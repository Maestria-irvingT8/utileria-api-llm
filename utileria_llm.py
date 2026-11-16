# Utileria de API call a un LLM

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import requests

CARPETA = Path(__file__).resolve().parent
ARCHIVO_ENV = CARPETA / ".env"
ARCHIVO_HISTORIAL = CARPETA / "historial.jsonl"
TIMEOUT = 120

# Groq, Gemini, OpenAI y Ollama usan el formato de OpenAI; Anthropic usa el suyo
PROVEEDORES = {
    "groq": {"formato": "openai", "url": "https://api.groq.com/openai/v1",
             "key": "GROQ_API_KEY", "modelo": "openai/gpt-oss-120b"},
    "gemini": {"formato": "openai", "url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key": "GEMINI_API_KEY", "modelo": "gemini-4-hydrogen"},
    "openai": {"formato": "openai", "url": "https://api.openai.com/v1",
               "key": "OPENAI_API_KEY", "modelo": "gpt-6-luna"},
    "anthropic": {"formato": "anthropic", "url": "https://api.anthropic.com/v1",
                  "key": "ANTHROPIC_API_KEY", "modelo": "claude-opus-5-5"},
    "ollama": {"formato": "openai", "url": "http://localhost:11434/v1",
               "key": None, "modelo": "llama3.2"},
}

ERRORES_HTTP = {
    400: "Peticion mal formada",
    401: "API key invalida",
    403: "Sin permiso para este recurso",
    404: "El modelo o la URL no existen",
    429: "Limite de peticiones alcanzado, espera un momento",
    500: "Error interno del servidor",
    503: "Servicio no disponible",
}


# Error propio para fallas de red, de HTTP o de formato de la respuesta
class ErrorAPI(Exception):
    pass


# Lee el archivo .env y guarda las API keys como variables de entorno
def cargar_env():
    if not ARCHIVO_ENV.exists():
        return
    for linea in ARCHIVO_ENV.read_text(encoding="utf-8-sig").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            nombre, valor = linea.split("=", 1)
            if valor.strip():
                os.environ.setdefault(nombre.strip(), valor.strip().strip("\"'"))


# Define las opciones que se pueden pasar desde la terminal
def leer_argumentos():
    parser = argparse.ArgumentParser(description="Utileria de API call a un LLM")
    parser.add_argument("prompt", nargs="*", help="pregunta (sin pregunta entra en modo chat)")
    parser.add_argument("-p", "--proveedor", choices=PROVEEDORES)
    parser.add_argument("-m", "--modelo")
    parser.add_argument("-s", "--sistema", help="instruccion de sistema")
    parser.add_argument("-t", "--temperatura", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--modelos", action="store_true", help="lista los modelos disponibles")
    parser.add_argument("--solo-respuesta", action="store_true", help="no muestra el trafico HTTP")
    return parser.parse_args()


# Elige proveedor, API key y modelo; termina si falta la key
def configurar(args):
    proveedor = (args.proveedor or os.environ.get("LLM_PROVEEDOR", "")).lower()
    if not proveedor:
        # usa el primer proveedor que tenga key en el .env
        proveedor = next((p for p, d in PROVEEDORES.items() if d["key"] and os.environ.get(d["key"])), "")
    if not proveedor:
        sys.exit("No hay ninguna API key. Agrega una en el archivo .env")
    if proveedor not in PROVEEDORES:
        sys.exit(f"Proveedor desconocido: {proveedor}")

    datos = PROVEEDORES[proveedor]
    api_key = os.environ.get(datos["key"], "") if datos["key"] else ""
    if datos["key"] and not api_key:
        sys.exit(f"Falta {datos['key']} en el archivo .env")

    temperatura = args.temperatura
    if temperatura is not None and datos["formato"] == "anthropic":
        print("Aviso: Claude no acepta temperature, se ignora")
        temperatura = None

    return {
        "proveedor": proveedor,
        "api_key": api_key,
        "modelo": args.modelo or os.environ.get("LLM_MODELO") or datos["modelo"],
        "sistema": args.sistema,
        "temperatura": temperatura,
        "max_tokens": args.max_tokens,
        "mostrar": not args.solo_respuesta,
    }


# Headers de autenticacion: cada proveedor pide la key de forma distinta
def headers_auth(conf):
    if PROVEEDORES[conf["proveedor"]]["formato"] == "anthropic":
        return {"x-api-key": conf["api_key"], "anthropic-version": "2026-10-21"}
    if conf["api_key"]:
        return {"Authorization": "Bearer " + conf["api_key"]}
    return {}


# Construye la URL, los headers y el cuerpo JSON de la peticion
def armar_peticion(conf, mensajes):
    datos = PROVEEDORES[conf["proveedor"]]
    headers = {"Content-Type": "application/json", **headers_auth(conf)}

    if datos["formato"] == "anthropic":
        url = datos["url"] + "/messages"
        cuerpo = {"model": conf["modelo"], "max_tokens": conf["max_tokens"] or 16000, "messages": mensajes}
        if conf["sistema"]:
            cuerpo["system"] = conf["sistema"]
    else:
        url = datos["url"] + "/chat/completions"
        cuerpo = {"model": conf["modelo"], "messages": mensajes}
        if conf["sistema"]:
            cuerpo["messages"] = [{"role": "system", "content": conf["sistema"]}] + mensajes
        if conf["temperatura"] is not None:
            cuerpo["temperature"] = conf["temperatura"]
        if conf["max_tokens"]:
            campo = "max_completion_tokens" if conf["proveedor"] == "openai" else "max_tokens"
            cuerpo[campo] = conf["max_tokens"]
    return url, headers, cuerpo


# Envia la peticion HTTP al servidor y regresa la respuesta y el tiempo que tardo
def enviar(metodo, url, headers, cuerpo=None, mostrar=True):
    if mostrar:
        mostrar_peticion(metodo, url, headers, cuerpo)
    inicio = time.time()
    try:
        respuesta = requests.request(metodo, url, headers=headers, json=cuerpo, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        raise ErrorAPI("El servidor no respondio a tiempo")
    except requests.exceptions.ConnectionError:
        raise ErrorAPI(f"No se pudo conectar con {url}")
    segundos = time.time() - inicio

    if mostrar:
        mostrar_respuesta(respuesta, segundos)
    if not respuesta.ok:
        raise ErrorAPI(mensaje_error(respuesta))
    return respuesta, segundos


# Arma un mensaje con el codigo de estado y el error que regreso el servidor
def mensaje_error(respuesta):
    codigo = respuesta.status_code
    try:
        datos = respuesta.json()
        if isinstance(datos, list):
            datos = datos[0]
        error = datos.get("error", datos)
        detalle = error.get("message", error) if isinstance(error, dict) else error
    except (ValueError, AttributeError, IndexError):
        detalle = respuesta.text[:300]
    return f"HTTP {codigo} - {ERRORES_HTTP.get(codigo, 'Error')}: {detalle}"


# Saca del JSON de respuesta el texto, los tokens usados y el motivo de fin
def leer_respuesta(formato, datos):
    try:
        if formato == "anthropic":
            texto = "".join(b["text"] for b in datos["content"] if b["type"] == "text")
            uso = datos.get("usage", {})
            return {"texto": texto, "fin": datos.get("stop_reason"),
                    "tokens_entrada": uso.get("input_tokens"), "tokens_salida": uso.get("output_tokens")}
        opcion = datos["choices"][0]
        uso = datos.get("usage", {})
        return {"texto": opcion["message"].get("content") or "", "fin": opcion.get("finish_reason"),
                "tokens_entrada": uso.get("prompt_tokens"), "tokens_salida": uso.get("completion_tokens")}
    except (KeyError, IndexError, TypeError):
        raise ErrorAPI("La respuesta del servidor no tiene el formato esperado")


# Muestra solo el inicio y el final de la API key
def ocultar_key(nombre, valor):
    if nombre.lower() not in ("authorization", "x-api-key"):
        return valor
    prefijo, _, key = valor.rpartition(" ")
    return f"{prefijo} {key[:4]}...{key[-4:]}".strip()


# Imprime lo que se envia: metodo, URL, headers y cuerpo
def mostrar_peticion(metodo, url, headers, cuerpo):
    print("\n--- PETICION ---")
    print(metodo, url)
    for nombre, valor in headers.items():
        print(f"{nombre}: {ocultar_key(nombre, valor)}")
    if cuerpo is not None:
        print(json.dumps(cuerpo, indent=2, ensure_ascii=False))


# Imprime lo que regresa el servidor: codigo de estado, headers y cuerpo
def mostrar_respuesta(respuesta, segundos):
    print(f"\n--- RESPUESTA ({respuesta.status_code} {respuesta.reason}, {segundos:.2f} s) ---")
    for nombre, valor in respuesta.headers.items():
        if nombre.lower() in ("content-type", "date", "retry-after") or "ratelimit" in nombre.lower():
            print(f"{nombre}: {valor}")
    try:
        print(json.dumps(respuesta.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(respuesta.text[:1000])


# Hace una consulta completa: arma, envia, lee la respuesta, la muestra y la guarda
def consultar(conf, mensajes):
    url, headers, cuerpo = armar_peticion(conf, mensajes)
    respuesta, segundos = enviar("POST", url, headers, cuerpo, conf["mostrar"])

    resultado = leer_respuesta(PROVEEDORES[conf["proveedor"]]["formato"], respuesta.json())
    resultado.update(proveedor=conf["proveedor"], modelo=conf["modelo"],
                     status=respuesta.status_code, segundos=round(segundos, 2))

    if conf["mostrar"]:
        print(f"\n--- RESULTADO ({resultado['modelo']}, "
              f"tokens: {resultado['tokens_entrada']} entrada / {resultado['tokens_salida']} salida) ---")
    # si el limite de tokens se acaba mientras el modelo razona, no llega texto
    print(resultado["texto"] or "(sin texto: el modelo uso todos los tokens)")
    if resultado["fin"] in ("length", "max_tokens"):
        print("\nAviso: la respuesta se corto por el limite de tokens")
    elif resultado["fin"] == "refusal":
        print("\nAviso: el modelo rechazo la peticion")

    guardar_historial(mensajes[-1]["content"], resultado)
    return resultado


# Agrega la consulta a historial.jsonl (un JSON por linea)
def guardar_historial(prompt, resultado):
    registro = {"fecha": datetime.now().isoformat(timespec="seconds"), "prompt": prompt, **resultado}
    registro["respuesta"] = registro.pop("texto")
    with open(ARCHIVO_HISTORIAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


# Peticion GET para obtener los modelos disponibles del proveedor
def listar_modelos(conf):
    url = PROVEEDORES[conf["proveedor"]]["url"] + "/models"
    respuesta, _ = enviar("GET", url, headers_auth(conf), None, conf["mostrar"])
    try:
        modelos = sorted(m["id"] for m in respuesta.json()["data"])
    except (KeyError, TypeError, ValueError):
        raise ErrorAPI("La respuesta no contiene una lista de modelos")
    print(f"\nModelos disponibles ({len(modelos)}):")
    print("\n".join(modelos))


# Conversacion de varios turnos. HTTP no guarda estado,
# Cada turno se manda toda la conversacion
def modo_chat(conf):
    print(f"Chat con {conf['modelo']} (escribe 'salir' para terminar)")
    mensajes = []
    while True:
        try:
            entrada = input("\nTu > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if entrada.lower() == "salir":
            break
        if not entrada:
            continue
        mensajes.append({"role": "user", "content": entrada})
        try:
            resultado = consultar(conf, mensajes)
            mensajes.append({"role": "assistant", "content": resultado["texto"]})
        except ErrorAPI as e:
            print(f"\nError: {e}", file=sys.stderr)
            mensajes.pop()


# Punto de entrada: lee la configuracion y decide que hacer
def main():
    # para que los acentos funcionen en la terminal de Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    cargar_env()
    args = leer_argumentos()
    conf = configurar(args)
    try:
        if args.modelos:
            listar_modelos(conf)
        elif args.prompt:
            consultar(conf, [{"role": "user", "content": " ".join(args.prompt)}])
        else:
            modo_chat(conf)
    except ErrorAPI as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

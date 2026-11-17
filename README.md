# Utilería de API call a un LLM

Programa en Python que funciona como cliente de un servicio web: se conecta a la API de un modelo de lenguaje (LLM), le envía una pregunta y recibe la respuesta generada.

## Qué hace

El programa arma una petición HTTP `POST` con la pregunta en formato JSON y la API key en los headers, y la envía al servidor del proveedor. El servidor procesa la solicitud con el modelo y regresa una respuesta JSON con el texto generado, los tokens usados y el código de estado.

Para que se vea el intercambio de mensajes entre las dos máquinas, el programa imprime:

- la petición enviada (método, URL, headers y cuerpo).
- la respuesta del servidor (código de estado, tiempo, headers y cuerpo).
- el resultado final (texto y tokens).

La API key se lee de un archivo `.env` (por ejemplo, `GROQ_API_KEY=tu_api_key`) para no dejarla en el código, y se muestra parcialmente oculta.

## Funciones

- Hacer una pregunta: `python utileria_llm.py "¿Qué es un web service?"`
- Modo chat de varios turnos: `python utileria_llm.py`. Como HTTP no guarda estado, en cada turno se reenvía toda la conversación.
- Listar los modelos disponibles con una petición `GET`: `python utileria_llm.py --modelos`
- Cambiar proveedor (`-p`), modelo (`-m`), instrucción de sistema (`-s`), temperatura (`-t`) o límite de tokens (`--max-tokens`).
- Mostrar solo el texto de la respuesta: `--solo-respuesta`
- Manejar errores comunes: key inválida (401), modelo inexistente (404), límite de peticiones (429), fallas del servidor (5xx) y falta de conexión.
- Guardar cada consulta en `historial.jsonl`.

Proveedores soportados: Groq, Gemini, OpenAI, Anthropic y Ollama (local). Por defecto usa Groq con el modelo `openai/gpt-oss-120b`.

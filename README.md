 
> Este bot te avisa, por Telegram o por registro de actividad, cuando un token **se dispara (pumped)**, **se desploma (rugged)** o aparece algún **evento relevante**. Ideal para *monitoring* simple sin cuadros complejos. 🧭

---

## 🧩 ¿Qué hace este bot?
- **Observa** precios y actividad de tokens en fuentes públicas tipo *Dexscreener*.
- **Detecta** movimientos fuertes en 24h (subidas o caídas) y otros eventos básicos.
- **Filtra ruido** (volumen poco creíble, supply dudoso, listas negras) para evitar falsas alarmas.
- **Avisa** por **Telegram** (opcional) con un botón que abre la ficha del par en Dexscreener.
- **Guarda** el historial de eventos para revisar más tarde (en archivos y en una base local).

> Piensa en él como un “radar” que te muestra **lo importante** y **a tiempo**. ⏱️

---

## 💎 Beneficios clave
- **Ahorro de tiempo:** no necesitas revisar paneles todo el día.
- **Alertas claras:** mensajes cortos con botón directo “Ver en Dexscreener”.  
- **Sin dependencia cloud:** todo corre **localmente** (tu equipo o servidor).  
- **Configurable sin tocar código:** ajustes en un archivo de configuración sencillo.
- **Historial propio:** te quedas con los datos (eventos) para auditoría o análisis.
 
---

## 🚦 Puesta en marcha (5 pasos)
1. **Descarga** este repositorio o el archivo del bot.  
2. **Crea** un archivo llamado `config.yaml` (ver ejemplo más abajo).  
3. (Opcional) **Activa Telegram**: indica el `bot_token` y el `chat_id` en `config.yaml`.  
4. **Inicia** el bot con doble clic o con `python bot.py`.  
5. **Listo** ✅: el bot empezará a revisar y a guardar/emitir alertas.

> Para **detener** el bot: cierra la ventana o presiona **Ctrl + C**. 🛑

---

## ✉️ Ejemplo de alerta (Telegram)

> **PUMPED**  
> **TOKEN_NAME**  
> Price: $0.012345  
> 24h: +58.20%  
> Vol: $120,000  
> CA: 0xABC…123  
> [🔎 Ver en Dexscreener]

*(Los valores son ilustrativos.)*
 
---

## 🎛️ Personalización (sin código)
Todo se ajusta en **`config.yaml`**. Los campos más usados:

- **Umbrales de alerta:** define a partir de qué **subida** o **caída** (24h) quieres avisos.  
  - *Ejemplo:* subir **≥ +50%** = “pumped”; caer **≤ −50%** = “rugged”.
- **Eventos que me importan:** puedes activar/desactivar tipos de evento (p. ej., solo “pumped”).
- **Listas negras:** tokens o desarrolladores que **no** quieres ver en alertas.
- **Frecuencia:** cada cuánto revisar (por defecto, cada **3 minutos** para el flujo rápido).
- **Telegram:** activar/desactivar avisos, cambiar destino, etc.

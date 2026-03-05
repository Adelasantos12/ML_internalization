# ML Internalization Model (LATAM)

Este repositorio ahora tiene dos componentes:

1. **Pipeline de modelado** para estimar capacidad de internalización del tratado.
2. **Web app interactiva** para explorar escenarios (ej. subir polarización y observar cambio en la probabilidad predicha).

## 1) Ejecutar pipeline

```bash
python treaty_model_pipeline.py
```

Genera:
- `outputs/model_results.json`
- `outputs/model_results.md`

## 2) Ejecutar web app local

```bash
python web_app.py
```

Abrir: `http://localhost:8000`

### ¿Qué permite la app?
- Mover sliders de variables clave (`polariz`, `checks`, `gov_seat_share`, `state_capacity_wgi`, `uhc`, `health_exp_gdp`).
- Ver la probabilidad estimada de mejora anual en SPAR.
- Ver curva de sensibilidad dinámica por variable.

## Deploy en Railway

Este proyecto no requiere dependencias externas para correr la web app.

- Railway Start Command: `python web_app.py`
- Railway usará automáticamente `PORT`.

Opcionalmente puedes usar `Procfile` incluido.


## Dependencias Python

- Se incluye `requirements.txt` en la raíz para compatibilidad con deploy platforms como Railway.
- La app usa solo librerías estándar de Python (sin paquetes externos obligatorios).

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


project_directory = Path(__file__).resolve().parent
model_path = project_directory / "pack_orders_model.joblib"

model_bundle = joblib.load(model_path)

model = model_bundle["model"]
features = model_bundle["features"]


app = FastAPI(
    title="Pack Orders Prediction API",
    version="1.0"
)


class OperationsInput(BaseModel):
    pick_units: float
    pick_lh: float
    put_units: float
    put_lh: float
    order_lh: float


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >
        <title>WDC Order TP Predictor</title>

        <style>
            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 30px;
                font-family: Arial, sans-serif;
                background:
                    linear-gradient(135deg, #0f172a, #1e3a5f);
                color: #1f2937;
            }

            .container {
                width: 100%;
                max-width: 650px;
                padding: 38px;
                border-radius: 18px;
                background: white;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
            }

            h1 {
                margin: 0 0 8px;
                color: #0f172a;
                text-align: center;
            }

            .subtitle {
                margin: 0 0 30px;
                color: #64748b;
                text-align: center;
            }

            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 18px;
            }

            .form-group {
                display: flex;
                flex-direction: column;
                gap: 7px;
            }

            label {
                font-size: 14px;
                font-weight: bold;
                color: #334155;
            }

            input {
                width: 100%;
                padding: 12px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                font-size: 16px;
            }

            input:focus {
                border-color: #2563eb;
                outline: 3px solid rgba(37, 99, 235, 0.15);
            }

            button {
                width: 100%;
                margin-top: 24px;
                padding: 14px;
                border: none;
                border-radius: 9px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            button:disabled {
                background: #94a3b8;
                cursor: wait;
            }

            #result {
                display: none;
                margin-top: 24px;
                padding: 22px;
                border-radius: 10px;
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                text-align: center;
            }

            #result.error {
                background: #fef2f2;
                border-color: #fecaca;
                color: #991b1b;
            }

            .prediction {
                margin-top: 6px;
                color: #1d4ed8;
                font-size: 32px;
                font-weight: bold;
            }

            .links {
                margin-top: 24px;
                text-align: center;
            }

            .links a {
                color: #2563eb;
                text-decoration: none;
            }

            @media (max-width: 600px) {
                .form-grid {
                    grid-template-columns: 1fr;
                }

                .container {
                    padding: 25px;
                }
            }
        </style>
    </head>

    <body>
        <main class="container">
            <h1>WDC Order TP Predictor</h1>

            <p class="subtitle">
                Enter estimated unit throughput and unit per hour productions rates for each area.
            </p>

            <form id="prediction-form">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="pick_units">Pick units</label>
                        <input
                            id="pick_units"
                            type="number"
                            step="any"
                            min="0"
                            required
                        >
                    </div>

                    <div class="form-group">
                        <label for="pick_lh">Pick rate (uph)</label>
                        <input
                            id="pick_lh"
                            type="number"
                            step="any"
                            min="0"
                            required
                        >
                    </div>

                    <div class="form-group">
                        <label for="put_units">Put units</label>
                        <input
                            id="put_units"
                            type="number"
                            step="any"
                            min="0"
                            required
                        >
                    </div>

                    <div class="form-group">
                        <label for="put_lh">Put rate (uph)</label>
                        <input
                            id="put_lh"
                            type="number"
                            step="any"
                            min="0"
                            required
                        >
                    </div>

                    <div class="form-group">
                        <label for="order_lh">Order rate (cph)</label>
                        <input
                            id="order_lh"
                            type="number"
                            step="any"
                            min="0"
                            required
                        >
                    </div>
                </div>

                <button id="submit-button" type="submit">
                    Predicted orders shipped
                </button>
            </form>

            <section id="result"></section>

            <div class="links">
                <a href="/docs">Open API documentation</a>
            </div>
        </main>

        <script>
            const form = document.getElementById("prediction-form");
            const result = document.getElementById("result");
            const button = document.getElementById("submit-button");

            form.addEventListener("submit", async (event) => {
                event.preventDefault();

                button.disabled = true;
                button.textContent = "Calculating...";

                result.classList.remove("error");
                result.style.display = "none";

                const inputData = {
                    pick_units: Number(
                        document.getElementById("pick_units").value
                    ),
                    pick_lh: Number(
                        document.getElementById("pick_lh").value
                    ),
                    put_units: Number(
                        document.getElementById("put_units").value
                    ),
                    put_lh: Number(
                        document.getElementById("put_lh").value
                    ),
                    order_lh: Number(
                        document.getElementById("order_lh").value
                    )
                };

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(inputData)
                    });

                    if (!response.ok) {
                        throw new Error(
                            "The server could not calculate a prediction."
                        );
                    }

                    const data = await response.json();

                    result.innerHTML = `
                        <div>Estimated packed orders</div>
                        <div class="prediction">
                            ${data.predicted_pack_orders.toLocaleString()}
                        </div>
                    `;

                    result.style.display = "block";
                } catch (error) {
                    result.textContent = error.message;
                    result.classList.add("error");
                    result.style.display = "block";
                } finally {
                    button.disabled = false;
                    button.textContent = "Predict packed orders";
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(data: OperationsInput):
    input_values = {
        "pick_units": data.pick_units,
        "pick_lh": data.pick_lh,
        "put_units": data.put_units,
        "put_lh": data.put_lh,
        "order_lh": data.order_lh
    }

    model_input = pd.DataFrame(
        [input_values],
        columns=features
    )

    predicted_orders = model.predict(model_input)[0]

    return {
        "predicted_pack_orders": round(
            float(predicted_orders),
            0
        )
    }
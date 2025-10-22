# nanny_decision_tree (no-SQL version)

Microservicio Python (Flask) que recibe filtros y la lista de niñeras desde Laravel y devuelve un ranking basado en reglas + DecisionTreeClassifier.

## Instalación
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Entrenar modelo
python -m scripts.train --csv path/to/export.csv
python -m scripts.train --json path/to/export.json

## Levantar API
export NANNY_API_KEY="mi_clave_secreta"
python -m api.app

## Uso (Laravel)
Enviar POST a /predict con JSON:
{
  "address": {"postal_code":"44220", "street":"La fabrica 962", "neighborhood":"Santa Elena"},
  "career":"Pedagogía",
  "courses":["Primeros Auxilios"],
  "qualities":["Paciente"],
  "availability": true,
  "nannies": [ /* listado de nannies desde Laravel */ ]
}

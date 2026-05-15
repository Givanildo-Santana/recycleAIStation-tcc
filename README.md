# RecycleAI-Station

Projeto de TCC desenvolvido na UNIP. Um sistema que classifica resíduos sólidos em tempo real usando visão computacional e controla uma esteira via Arduino para separar os materiais automaticamente.

## Tecnologias

Python, PySide6, PyTorch (YOLOv5/YOLOv8), OpenCV, SQLite, Arduino

## Requisitos

- Python 3.10+
- Arduino Uno com o firmware carregado
- Modelo treinado em TorchScript (.pt) registrado pelo painel de manutenção

## Estrutura

- `app/` — entrypoint
- `core/` — inferência e hardware
- `db/` — banco e migrações
- `gui/` — interface
- `firmware/` — código Arduino
- `tests/` — testes de integração
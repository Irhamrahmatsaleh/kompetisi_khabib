# kompetisi_khabib

Multi-asset demo competition scanner for FX, metals, indices, oil, and crypto market proxies. It ranks candidates by trend, momentum, volatility, and ATR expansion, then prints BUY/SELL/WAIT scenarios with Entry, Stop Loss, TP1, TP2, market-status estimates, and competition planning math. The output is demo-planning context only, not a guarantee.

1. Clone the repository on Ubuntu:
```bash
git clone https://github.com/Irhamrahmatsaleh/kompetisi_khabib.git
cd kompetisi_khabib
```

2. Install the dependencies:
```bash
sudo apt update && sudo apt install -y python3 python3-full python3-venv make
make install
```

3. Execute the engine:
```bash
make serve
```

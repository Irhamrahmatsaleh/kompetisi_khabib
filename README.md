# kompetisi_khabib

Broad demo-competition scanner for FX, metals, energy, indices, crypto, stocks, and ETFs. It compares a wide market universe, ranks BUY/SELL candidates, and prints Entry, Stop Loss, TP1, TP2, score, risk regime, and market-status estimates. The output is demo-planning context only, not a guarantee; the trading platform's BUY/SELL buttons, spread, margin, and contract size remain the final truth.

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

3. Execute the broad scanner:
```bash
make serve
```

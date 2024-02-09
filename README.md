# sbrc2024

- Run `new_install.sh` to install and setup pyenv. Follow the echoed instructions to setup a virtual environment and install requirements.

## Notes

- Each simulation example is accompanied by a script to read its log files and plot its results. The plots will only work if the values in the simulations are used as is. For example: the centralized scenario runs for 10 epochs over 6 different combinations of batch size and learning rate, it also uses a verbose level of 2.

- `pgrep python | xargs kill` can be used to kill the server and the clients.


## This is because I can't remember screen commands:

- screen -S name
- CTRL+A D (dettach)
- screen -ls
- screen -r name (attach)

# BibTex

TBD

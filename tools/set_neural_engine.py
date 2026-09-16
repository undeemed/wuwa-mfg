# SPDX-License-Identifier: Apache-2.0
"""Request a live engine switch, or reload, through the existing OptiScaler NR path."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuwa_mfg.core import atomic_write,find_game

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('engine',choices=('student','nvidia','reload'))
    p.add_argument('--game',required=True,type=Path)
    a=p.parse_args();game=find_game(a.game)
    atomic_write(game/'neural-engine.request',(a.engine+'\n').encode())
    print('Request saved; the next active neural frame will consume it. F8 enables/disables the effect.')

if __name__=='__main__':main()

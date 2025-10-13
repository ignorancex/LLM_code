import cv2
import sys

shutter=['1/4000', '1/2000', '1/1000', '1/500', '1/250', '1/125', '1/60', '1/30', '1/15', '1/8', '1/4', '0.5', '1', '2', '4', '8', '15', '30']
iso=[100, 200, 400, 800, 1600, 3200, 6400]
fnumber=[22, 16, 11, 8, 5.6]

def get_ev_from_auto(auto_shutter, auto_iso, auto_fnumber):
    s_0 = min(range(len(shutter)), key=lambda i: abs(eval(shutter[i]) - eval(auto_shutter)))
    i_0 = min(range(len(iso)), key=lambda i: abs(iso[i] - auto_iso))
    f_0 = min(range(len(fnumber)), key=lambda i: abs(fnumber[i] - auto_fnumber))
    print(s_0, i_0, f_0)

if __name__ == '__main__':
    auto_shutter = sys.argv[1]
    auto_iso = int(sys.argv[2])
    auto_fnumber = float(sys.argv[3])
    get_ev_from_auto(auto_shutter, auto_iso, auto_fnumber)

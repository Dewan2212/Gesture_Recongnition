# main.py — ESP32 MicroPython (Hybrid: MCU sampling/normalization, PC GUI classifies)
# Commands over serial (from your GUI): hdr | cal2 | start | stop | probe <1..3> | dbg | help
#
# PHYSICAL FLEX WIRING (as you confirmed):
#   F1 = 34 (Middle), F2 = 32 (Index), F3 = 33 (Thumb)
# We map PHYS(F1,F2,F3) -> LOG(Thumb,Index,Middle) using MAP_PHYS_TO_LOG = [2,1,0]

import sys, time, math
import machine

USER_ID = "user1"

# ---------- Pins ----------
ADC_PINS = [34, 32, 33]   # F1=Middle(34), F2=Index(32), F3=Thumb(33)
MAP_PHYS_TO_LOG = [2, 1, 0]  # [Thumb, Index, Middle] = phys[2], phys[1], phys[0]

I2C_SCL, I2C_SDA = 22, 21
I2C_FREQ = 400_000

# ---------- Sampling ----------
TARGET_HZ   = 80
OS_SAMPLES  = 16   # oversample per ADC read for smoother values

# ---------- Calibration & normalization ----------
DEAD_MIN_DELTA = 10
RAIL_NEAR      = 35
MIN_RANGE      = 20

# ---------- IMU (MPU6050 minimal) ----------
class MPU6050:
    ADDR=0x68; PWR=0x6B; AX=0x3B; GX=0x43; AC=0x1C; GC=0x1B
    def __init__(self,i2c):
        self.i2c=i2c
        try:
            self.i2c.writeto_mem(self.ADDR,self.PWR,b'\x00')   # wake
            self.i2c.writeto_mem(self.ADDR,self.AC,b'\x00')    # ±2g
            self.i2c.writeto_mem(self.ADDR,self.GC,b'\x00')    # ±250 dps
        except: pass
    def _r16s(self,reg):
        try:
            b=self.i2c.readfrom_mem(self.ADDR,reg,2)
            v=(b[0]<<8)|b[1]
            return v-65536 if v&0x8000 else v
        except:
            return 0
    def accel(self):
        ax=self._r16s(self.AX)/16384.0; ay=self._r16s(self.AX+2)/16384.0; az=self._r16s(self.AX+4)/16384.0
        return ax,ay,az
    def gyro(self):
        gx=self._r16s(self.GX)/131.0; gy=self._r16s(self.GX+2)/131.0; gz=self._r16s(self.GX+4)/131.0
        return gx,gy,gz

def accel_to_rp(ax,ay,az):
    roll  = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay+az*az)))
    return roll, pitch

# ---------- ADC setup ----------
try:
    ATTN_OPTS = [
        ("0dB",   machine.ADC.ATTN_0DB),
        ("2.5dB", machine.ADC.ATTN_2_5DB),
        ("6dB",   machine.ADC.ATTN_6DB),
        ("11dB",  machine.ADC.ATTN_11DB),
    ]
except:
    ATTN_OPTS = [("11dB", machine.ADC.ATTN_11DB)]

adcs=[]
for p in ADC_PINS:
    a=machine.ADC(machine.Pin(p))
    try: a.width(machine.ADC.WIDTH_12BIT)
    except: pass
    try: a.atten(ATTN_OPTS[-1][1])
    except: pass
    adcs.append(a)

adc_attn_idx = [len(ATTN_OPTS)-1]*3

# ---------- Globals ----------
i2c  = machine.SoftI2C(scl=machine.Pin(I2C_SCL), sda=machine.Pin(I2C_SDA), freq=I2C_FREQ)
imu  = MPU6050(i2c)

def now_ms(): return time.ticks_ms()

def read_adc_os(ch):
    s=0
    for _ in range(OS_SAMPLES):
        s += adcs[ch].read()
    return s//OS_SAMPLES

def read_raw_3():
    return [read_adc_os(0), read_adc_os(1), read_adc_os(2)]

# calibration data per PHYSICAL channel
open_raw  = [0,0,0]
fist_raw  = [0,0,0]
pol       = [ 1, 1, 1 ]
rng_raw   = [300,300,300]
gain_inv  = [1/300.0, 1/300.0, 1/300.0]
dead_flags=[False,False,False]
adc_attn_idx=[len(ATTN_OPTS)-1]*3

# IMU
roll_f=0.0; pitch_f=0.0
rp_zero=[0.0,0.0]
last_imu_ms=now_ms()

# stream
recording=False
seq_i=0

# ---------- helpers ----------
def clamp01(x):
    if x<0: return 0.0
    if x>1: return 1.0
    return x

def phys_norm(fr_phys):
    out=[0,0,0]
    for i in range(3):
        dv = (fr_phys[i] - open_raw[i]) * pol[i]
        out[i] = int(clamp01(dv * gain_inv[i]) * 1000 + 0.5)
    return out

def to_logical(arr_phys):
    return [arr_phys[MAP_PHYS_TO_LOG[0]],
            arr_phys[MAP_PHYS_TO_LOG[1]],
            arr_phys[MAP_PHYS_TO_LOG[2]]]

def print_header():
    print("i,ts_ms,user,raw_F1,raw_F2,raw_F3,nT,nI,nM,bitsTIM,roll,pitch,g_auto,g_manual,mode")

# ---------- stdin (nonblocking) ----------
def stdin_ready():
    try:
        import uselect
        p=uselect.poll()
        p.register(sys.stdin, uselect.POLLIN)
        return bool(p.poll(0))
    except:
        return False

# ---------- calibration ----------
def avg_short(sec=0.35):
    n=max(10,int(sec*TARGET_HZ))
    s=[0,0,0]
    for _ in range(n):
        r=read_raw_3()
        s[0]+=r[0]; s[1]+=r[1]; s[2]+=r[2]
        time.sleep(1.0/max(TARGET_HZ,1))
    return [s[0]//n, s[1]//n, s[2]//n]

def do_cal2():
    global open_raw, fist_raw, pol, rng_raw, gain_inv, adc_attn_idx, dead_flags, rp_zero, roll_f, pitch_f

    print('{"note":"cal2_start","map_phys_to_log":%s}'%str(MAP_PHYS_TO_LOG))
    print(">> Hold OPEN (hand flat). Starting in 2s…")
    time.sleep(2.0)

    open_by_attn=[]
    for name,att in ATTN_OPTS:
        for i in range(3):
            try: adcs[i].atten(att)
            except: pass
        open_by_attn.append(avg_short(0.35))

    ax,ay,az=imu.accel()
    rr,pp=accel_to_rp(ax,ay,az)
    rp_zero=[rr,pp]; roll_f=0.0; pitch_f=0.0

    print(">> Now make a FIST (bend all three). Starting in 2s…")
    time.sleep(2.0)

    fist_by_attn=[]
    for name,att in ATTN_OPTS:
        for i in range(3):
            try: adcs[i].atten(att)
            except: pass
        fist_by_attn.append(avg_short(0.35))

    chosen_open=[0,0,0]; chosen_fist=[0,0,0]
    for ch in range(3):
        best_j=0; best_score=-1e9; bo=0; bf=0
        for j,(name,att) in enumerate(ATTN_OPTS):
            o=open_by_attn[j][ch]; f=fist_by_attn[j][ch]
            delta=abs(f-o)
            rail_pen=0
            if o<RAIL_NEAR or f<RAIL_NEAR: rail_pen += (RAIL_NEAR - min(o,f))
            if o>4095-RAIL_NEAR or f>4095-RAIL_NEAR: rail_pen += (max(o,f) - (4095-RAIL_NEAR))
            score=delta - rail_pen
            if score>best_score:
                best_score=score; best_j=j; bo=o; bf=f
        adc_attn_idx[ch]=best_j; chosen_open[ch]=bo; chosen_fist[ch]=bf
        try: adcs[ch].atten(ATTN_OPTS[best_j][1])
        except: pass

    open_raw = chosen_open[:]
    fist_raw = chosen_fist[:]

    dead_flags=[False,False,False]
    for ch in range(3):
        delta = fist_raw[ch] - open_raw[ch]
        pol[ch] = 1 if delta>=0 else -1
        rng_raw[ch] = max(abs(delta), MIN_RANGE)
        gain_inv[ch] = 1.0/float(rng_raw[ch])
        if abs(delta) < DEAD_MIN_DELTA:
            dead_flags[ch]=True

    diag = {
        "ok":"cal2",
        "pins":ADC_PINS,
        "open_raw":open_raw,
        "fist_raw":fist_raw,
        "pol":pol,
        "range":rng_raw,
        "atten":[ATTN_OPTS[i][0] for i in adc_attn_idx],
        "dead_phys":[i+1 for i,x in enumerate(dead_flags) if x]
    }
    print(diag)

    # If Thumb (physical 3) is dead, print hints
    thumb_phys=2
    if dead_flags[thumb_phys]:
        print('{"warn":"thumb_channel_flat","phys":"F3","pin":%d}'%ADC_PINS[thumb_phys])

# ---------- utilities ----------
def dbg():
    print({
        "open_raw":open_raw, "fist_raw":fist_raw, "pol":pol, "range":rng_raw,
        "atten":[ATTN_OPTS[i][0] for i in adc_attn_idx],
        "dead_phys":[i+1 for i,x in enumerate(dead_flags) if x],
        "map_phys_to_log":MAP_PHYS_TO_LOG
    })

def probe(ch):
    ch=int(ch)-1
    if ch<0 or ch>2:
        print('{"err":"probe","msg":"channel must be 1..3"}'); return
    print('{"probe":"start","phys":%d,"pin":%d}'%(ch+1, ADC_PINS[ch]))
    t0=now_ms()
    while time.ticks_diff(now_ms(),t0)<8000:
        r=read_adc_os(ch)
        print("raw_F%d=%d"%(ch+1,r))
        time.sleep(0.06)
    print('{"probe":"done"}')

def print_help():
    print("Commands: hdr | cal2 | start | stop | probe <1..3> | dbg | help")

def handle_cmd(line):
    parts=line.strip().split()
    if not parts: return
    c=parts[0].lower()
    global recording
    if c=="hdr":    print_header()
    elif c=="start": recording=True;  print('{"ok":"start"}')
    elif c=="stop":  recording=False; print('{"ok":"stop"}')
    elif c=="cal2":  do_cal2()
    elif c=="probe" and len(parts)==2:
        try: probe(int(parts[1]))
        except: print('{"err":"probe"}')
    elif c=="dbg":  dbg()
    elif c=="help": print_help()

# ---------- boot banner ----------
print('READY. PHYS(F1=34 Middle, F2=32 Index, F3=33 Thumb) -> LOG(Thumb,Index,Middle) via', MAP_PHYS_TO_LOG)
print_header()
print('Tip: from GUI click Calibrate (cal2) then Start.')

# ---------- main loop ----------
dt_target=1.0/max(TARGET_HZ,1)

while True:
    # commands
    if stdin_ready():
        try:
            handle_cmd(sys.stdin.readline())
        except: pass

    t0=now_ms()

    # sensor reads
    fr_phys = read_raw_3()
    nv_phys = phys_norm(fr_phys)
    nv_log  = to_logical(nv_phys)

    # IMU update (complementary)
    ax,ay,az = imu.accel(); gx,gy,gz = imu.gyro()
    nowi=now_ms(); dt=max(0.001,(time.ticks_diff(nowi,last_imu_ms))/1000.0); last_imu_ms=nowi
    rr,pp = accel_to_rp(ax,ay,az)
    roll_f  = 0.96*( (roll_f + gx*dt) ) + 0.04*(rr - rp_zero[0])
    pitch_f = 0.96*( (pitch_f + gy*dt) ) + 0.04*(pp - rp_zero[1])

    # stream
    if recording:
        seq_i += 1
        print("{i},{ts},{uid},{r1},{r2},{r3},{nT},{nI},{nM},{bits},{r:.2f},{p:.2f},{ga},{gm},{mode}".format(
            i=seq_i, ts=now_ms(), uid=USER_ID,
            r1=fr_phys[0], r2=fr_phys[1], r3=fr_phys[2],
            nT=nv_log[0], nI=nv_log[1], nM=nv_log[2],
            bits="000", r=roll_f, p=pitch_f, ga=-1, gm=-1, mode="auto"
        ))

    # pace
    elapsed = time.ticks_diff(now_ms(), t0)/1000.0
    if elapsed<dt_target:
        time.sleep(dt_target - elapsed)

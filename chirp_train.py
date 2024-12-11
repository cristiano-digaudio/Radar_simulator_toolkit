import numpy as np 

def chirp_pulse_train(f0, f1, pulse_duration, prf, fs, num_pulses):
    """
    Crea un treno di impulsi chirp.

    Parametri:
    - f0: frequenza iniziale del chirp (Hz)
    - f1: frequenza finale del chirp (Hz)
    - pulse_duration: durata di ciascun impulso chirp (s)
    - prf: frequenza di ripetizione degli impulsi (Hz)
    - fs: frequenza di campionamento (Hz)
    - num_pulses: numero di impulsi chirp nel treno

    Restituisce:
    - segnali: array che rappresenta il treno di impulsi chirp
    - t: array temporale per il treno di impulsi
    """
    # Durata totale del segnale
    duration = (num_pulses - 1) / prf + pulse_duration
    t = np.arange(0, duration, 1/fs)  # Asse temporale totale
    
    # Coefficiente del chirp
    beta = (f1 - f0) / pulse_duration

    # Array per il segnale complessivo
    segnali = np.zeros_like(t, dtype=complex)

    # Generazione degli impulsi chirp
    for i in range(num_pulses):
        start_time = i / prf  # Inizio del chirp
        end_time = start_time + pulse_duration  # Fine del chirp
        
        # Indici del chirp
        chirp_idx = (t >= start_time) & (t < end_time)
        t_chirp = t[chirp_idx] - start_time  # Tempo relativo per il chirp
        
        # Creazione del singolo impulso chirp
        segnali[chirp_idx] = np.exp(1j * 2 * np.pi * (f0 * t_chirp + 0.5 * beta * t_chirp**2))
    
    return segnali, t
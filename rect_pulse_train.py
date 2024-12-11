import numpy as np 
def rect_pulse_train(pulse_duration, prf, fs, num_pulses, amplitude=1):
        """
        Crea un treno di impulsi rettangolari.

        Parametri:
        - pulse_duration: durata di ciascun impulso rettangolare 
        - prf: frequenza di ripetizione degli impulsi (PRF) (Hz)
        - fs: frequenza di campionamento (Hz)
        - num_pulses: numero di impulsi rettangolari nel treno
        - amplitude: ampiezza degli impulsi rettangolari

        Restituisce:
        - segnali: array che rappresenta il treno di impulsi rettangolari
        - t: array temporale per il treno di impulsi
        """
        duration = num_pulses / prf  # Durata complessiva del segnale
        t = np.arange(0, (duration), 1/fs)  # Tempo totale

        # Creazione di un treno di impulsi rettangolari
        pulse_times = np.arange(0, num_pulses) / prf  # I tempi di inizio di ogni impulso
        
        # Array per il segnale
        segnali = np.zeros_like(t)
        
        # Creazione degli impulsi rettangolari
        for pulse_time in pulse_times:
            start_idx = int(pulse_time * fs)  # Indice di inizio del rettangolo
            end_idx = int((pulse_time + pulse_duration) * fs)  # Indice di fine del rettangolo
            segnali[start_idx:end_idx] = amplitude  # Imposta l'ampiezza del rettangolo
        
        return segnali, t

import numpy as np

def rect_pulse_train_stagg_prf(pulse_duration, prf_values, prf_counts, fs, total_pulses, amplitude=1):
    """
    Crea un treno di impulsi rettangolari con PRF alternate, continuando a seguire il pattern definito 
    se il numero di impulsi supera il conteggio dato da prf_counts.

    Parametri:
    - pulse_duration: durata di ciascun impulso rettangolare (s)
    - prf_values: lista di valori PRF (Hz)
    - prf_counts: lista di numeri di impulsi consecutivi per ogni PRF
    - fs: frequenza di campionamento (Hz)
    - total_pulses: numero totale di impulsi
    - amplitude: ampiezza degli impulsi rettangolari

    Restituisce:
    - segnali: array che rappresenta il treno di impulsi rettangolari
    - t: array temporale per il treno di impulsi
    """
    if len(prf_values) != len(prf_counts):
        raise ValueError("prf_values e prf_counts devono avere la stessa lunghezza.")

    # Creazione della sequenza PRF alternata
    prf_sequence = []
    while len(prf_sequence) < total_pulses:
        for prf, count in zip(prf_values, prf_counts):
            prf_sequence.extend([prf] * count)
            if len(prf_sequence) >= total_pulses:
                break

    # Troncamento della sequenza PRF al numero totale di impulsi
    prf_sequence = prf_sequence[:total_pulses]

    # Inizializzazione dei parametri temporali
    total_duration = 0  # Durata totale iniziale
    pulse_times = [0]  # Lista per i tempi di inizio degli impulsi

    for prf in prf_sequence:
        # Calcola l'intervallo tra gli impulsi
        pulse_interval = 1 / prf
        total_duration += pulse_interval
        pulse_times.append(total_duration)

    total_duration += pulse_duration  # Aggiungiamo la durata dell'ultimo impulso
    t = np.arange(0, total_duration, 1 / fs)  # Array temporale totale
    segnali = np.zeros_like(t)  # Array per il segnale

    # Creazione degli impulsi rettangolari
    for pulse_time in pulse_times[:-1]:  # Escludiamo l'ultimo punto oltre il segnale
        start_idx = int(pulse_time * fs)  # Indice di inizio del rettangolo
        end_idx = int(min((pulse_time + pulse_duration) * fs, len(t)))  # Indice di fine del rettangolo
        segnali[start_idx:end_idx] = amplitude  # Imposta l'ampiezza del rettangolo

    return segnali, t







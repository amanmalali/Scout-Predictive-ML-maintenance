import numpy as np

def generate_exponential_decay_array(length, decay_factor):
    """
    Generate an array with values that start at 1 and decay exponentially.

    Parameters:
    length (int): The length of the array.
    decay_factor (float): The decay factor. Higher values lead to faster decay.

    Returns:
    numpy.ndarray: Array of exponentially decaying values.
    """
    # Generate an array of indices from 0 to length-1
    indices = np.arange(length)
    
    # Compute the exponential decay
    decay_array = np.exp(-decay_factor * indices)
    
    return decay_array


def combine_xy_with_transition(x1, y1, x2, y2, transition_window):
    """
    Combine (x1, y1) and (x2, y2) into a single stream of length len(x1)+len(x2)
    with a linear transition from dataset 1 to dataset 2 over `transition_window`
    steps. At each step, picks from dataset 1 with probability p(i) that ramps 1→0
    across the transition window; falls back to the other dataset if the chosen one
    is exhausted.

    Args:
        x1, y1: First dataset features/labels.
        x2, y2: Second dataset features/labels.
        transition_window: Transition length (steps) for probability ramp.

    Returns:
        sim_x, sim_y: Combined stream as NumPy arrays.
    """

    
    window_probs = np.linspace(1, 0, transition_window)

    transition_start=int(len(y1)-transition_window//2)

    probs=np.ones(transition_start)
    probs=np.concatenate([probs,window_probs],axis=0)
    drift_probs=np.zeros(len(y1)+len(y2)-len(probs))
    probs=np.concatenate([probs,drift_probs],axis=0)
    sim_x=[]
    sim_y=[]
    train_i=0
    test_i=0
    drifted_i=0
    print("LENGTHS :",len(probs))
    for i in range(len(x1)+len(x2)):
        pick=np.random.choice([0,1],1,p=[probs[i],1-probs[i]])
        if pick==0:
            if train_i<len(x1):
                sim_x.append(x1[train_i])
                sim_y.append(y1[train_i])
                train_i+=1
            elif drifted_i<len(x2):
                sim_x.append(x2[drifted_i])
                sim_y.append(y2[drifted_i])
                drifted_i+=1
        elif pick==1:
            if drifted_i<len(x2):
                sim_x.append(x2[drifted_i])
                sim_y.append(y2[drifted_i])
                drifted_i+=1
            elif train_i<len(x1):
                sim_x.append(x1[train_i])
                sim_y.append(y1[train_i])
                train_i+=1
    
    return np.array(sim_x),np.array(sim_y)

def add_chunks_periodically_separate(a_x, a_y, b_x, b_y, chunk_size, period):
    """
    Build a stream by alternating consecutive chunks from A and slices from B:
    append `chunk_size` samples from (a_x,a_y), then `period` samples from (b_x,b_y),
    repeating until A is consumed; finally append any remaining B.

    Args:
        a_x, a_y: Dataset A features/labels.
        b_x, b_y: Dataset B features/labels.
        chunk_size: Number of A samples per chunk.
        period: Number of B samples inserted after each A chunk.

    Returns:
        sim_x, sim_y: Combined stream as NumPy arrays.
    """
    a_start=0
    b_start=0
    a_end=chunk_size
    b_end=period
    chunks=len(a_x)//chunk_size
    sim_x=a_x[a_start:a_end]
    sim_y=a_y[a_start:a_end]
    for c in range(chunks):
        if c==0:
            sim_x=a_x[a_start:a_end]
            sim_y=a_y[a_start:a_end]
        else:
            sim_x=np.concatenate([sim_x,a_x[a_start:a_end]])
            sim_y=np.concatenate([sim_y,a_y[a_start:a_end]])
        sim_x=np.concatenate([sim_x,b_x[b_start:b_end]])
        sim_y=np.concatenate([sim_y,b_y[b_start:b_end]])

        a_start=a_end
        a_end=a_end+chunk_size

        b_start=b_end
        b_end=b_end+period

        if a_end>len(a_x):
            a_end=len(a_x)
        
        if b_end>len(b_x):
            b_end=len(b_x)

    if b_end<len(b_x):
        b_end=len(b_x)
    sim_x=np.concatenate([sim_x,b_x[b_start:b_end]])
    sim_y=np.concatenate([sim_y,b_y[b_start:b_end]])
    return sim_x,sim_y


def simulate_stream_with_drift(train_x, train_y, drifted_x, drifted_y, decay=0.0001, rng=None):
    """
    Mixes train and drifted samples into a single simulated stream using
    exponentially-decaying probability of picking train vs drifted over time.

    Args:
        train_x, train_y: arrays/lists of training samples and labels
        drifted_x, drifted_y: arrays/lists of drifted samples and labels
        decay: decay rate passed to generate_exponential_decay_array
        rng: optional numpy random generator (np.random.default_rng()) for reproducibility

    Returns:
        sim_x, sim_y: numpy arrays of the simulated stream
    """
    if rng is None:
        rng = np.random.default_rng()

    total_len = len(train_x) + len(drifted_x)
    probs = generate_exponential_decay_array(total_len, decay)

    sim_x = []
    sim_y = []
    train_i = 0
    drifted_i = 0

    for i in range(total_len):
        # pick 0 => prefer train, pick 1 => prefer drifted
        pick = rng.choice([0, 1], p=[probs[i], 1 - probs[i]])

        if pick == 0:
            if train_i < len(train_x):
                sim_x.append(train_x[train_i])
                sim_y.append(train_y[train_i])
                train_i += 1
            elif drifted_i < len(drifted_x):
                sim_x.append(drifted_x[drifted_i])
                sim_y.append(drifted_y[drifted_i])
                drifted_i += 1
        else:  # pick == 1
            if drifted_i < len(drifted_x):
                sim_x.append(drifted_x[drifted_i])
                sim_y.append(drifted_y[drifted_i])
                drifted_i += 1
            elif train_i < len(train_x):
                sim_x.append(train_x[train_i])
                sim_y.append(train_y[train_i])
                train_i += 1

    return np.array(sim_x), np.array(sim_y)


def simulate_no_drift_stream(train_x, train_y, drifted_x, drifted_y, rng=None):
    """
    Builds a simulated stream of length len(train_x)+len(drifted_x) by starting with
    (train_x, train_y) and then repeatedly appending shuffled copies of (train_x, train_y)
    until reaching the target length, then truncating.

    Args:
        train_x, train_y: base arrays/lists for the beginning of the stream
        drifted_x, drifted_y: only used to determine target stream length
        rng: optional numpy random generator (np.random.default_rng()) for reproducibility

    Returns:
        sim_x, sim_y: numpy arrays of length len(train_x)+len(drifted_x)
    """
    if rng is None:
        rng = np.random.default_rng()

    new_len = len(train_x) + len(drifted_x)
    train_len = len(train_x)
    reps = new_len // train_len

    sim_x = np.array(train_x)
    sim_y = np.array(train_y)

    for _ in range(reps):
        shuffled_idx = rng.choice(np.arange(train_len), train_len, replace=False)
        new_sim_x = np.array(train_x)[shuffled_idx]
        new_sim_y = np.array(train_y)[shuffled_idx]
        sim_x = np.concatenate([sim_x, new_sim_x], axis=0)
        sim_y = np.concatenate([sim_y, new_sim_y], axis=0)

    sim_x = sim_x[:new_len]
    sim_y = sim_y[:new_len]
    return sim_x, sim_y


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

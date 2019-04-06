import numpy as np
import matplotlib.pyplot as plt
import RNA

def seq_sim(seq0,seq1):
    return sum(1 for a, b in zip(seq0, seq1) if a != b)
replace_alpha = {'C':['U','G','A'],
                 'U':['G','A','C'],
                 'G':['A','C','U'],
                 'A':['C','U','G']}
def gen_n_mutations(orig, n=10):
    seq = list(orig)
    mutation_list = []
    for i in np.sort(np.random.choice(np.arange(len(orig)),replace=False,size=n)):
        seq[i]  = np.random.choice(replace_alpha[seq[i]])
        mutation_list.append(''.join([orig[i],str(i),seq[i]]))
    
    return ''.join(x for x in seq),np.array(mutation_list)

def count_differences(seq1,seq2):
    return sum(1 for a, b in zip(seq1, seq2) if a != b)
def do_point_mutation(orig_seq, mutation):
    """
    do -> defined mutation
    gen -> random mutation
    """
    if mutation == '' or mutation is None:
        #don't actually perform a mutation
        #stay as wt
        print('here')
        return orig_seq
    idx = np.int(mutation[1:-1])
    return orig_seq[:idx]+mutation[-1]+orig_seq[idx+1:]
def do_mut_list(orig_seq, mut_list):
    """
    inputs
    ------
    mut_list : list of strings
                mutations of form 'baselocationbase' eg U30C
    """
    seq = ''+orig_seq
    for m in range(len(mut_list)):
        seq = do_point_mutation(seq,mut_list[m])
    return seq
def do_n_mutation(orig_seq, mut_list, which=None):
    """
    inputs
    ------
    mut_list : list of strings
                mutations of form 'baselocationbase' eg U30C
    which : iterable, integer, None, 'all','first,'last'
                indices of mut_list to perform
                None -> all
                'all' -> all
                'first' -> first
                'last' -> last
    """
    if which is None or which=='all':
        which = np.arange(len(mut_list))
    elif which=='first':
        which = [0]
    elif which =='last':
        which = [-1]
    else:
        which = np.atleast_1d(which)
    seq = ''+orig_seq
    for m in which:
        seq = do_point_mutation(seq,mut_list[m])
    return seq



letters_2_num = {'C':0,'U':1,'G':2,'A':3}
def str2num(seq):
    return [letters_2_num[character] for character in seq]


def imshow_masked(arr,cbar=True,offset = -1):
    """
    imshow with the lower triangle masked
    """
    n = arr.shape[0]
    idx = np.tril_indices(n,k=offset)
    mask_arr =np.ones([n,n])
    mask_arr[idx[0],idx[1]]=0
#     mask_arr[0,0]=1
    a = np.ma.masked_where(mask_arr==0, arr)
    plt.imshow(a)
    if cbar:
        plt.colorbar()
        
        
def gen_G(n = 1):
    G = np.atleast_1d(1)
    for order in range(1,n+1):
        G = np.block([[G,np.zeros([G.shape[0],G.shape[0]])],[-G,G]])    
    return G
def gen_H(n = 1):
    H = np.atleast_1d(1)
    for order in range(1,n+1):
        H = np.block([[H,H],[H,-H]])    
    return H
def gen_V(n = 1):
    V = np.atleast_1d(1)
    for order in range(1,n+1):
        shp = [V.shape[0],V.shape[0]]
        V = np.block([[.5*V,np.zeros(shp)],[np.zeros(shp),-V]])
    return V

def calc_bkd_epi(y,N=None):
    if N is None:
        N = np.int(np.log2(y.shape[0]))
    V = gen_V(N)
    H = gen_H(N)
    return V @ H @ y
def calc_ref_epi(y,N=None):
    if N is None:
        N = np.int(np.log2(y.shape[0]))
    G = gen_G(N)
    
    return G @ y

def get_N_nonzero(seq,mut_list,threshold = .2):
    mut_list = np.asanyarray(mut_list)
    N = len(mut_list)
    for i in range(2**N):
        idx = np.array(list(base_str.format(i)),dtype=np.int).astype(np.bool)
        y[i] = RNA.fold(do_mut_list(seq,mut_list[idx]))[-1]
    return np.sum(np.abs(ref_epi(y))<threshold),np.sum(np.abs(    bkgd_epi(y))<threshold)
def calc_phenotype_old(base_seq,mut_list):
    N = len(mut_list)
    base_str = '{:0'+str(N)+'b}'
    y = np.zeros(2**N)
    for i in range(2**N):
        idx = np.array(list(base_str.format(i)),dtype=np.int).astype(np.bool)
        y[i] = RNA.fold(do_mut_list(base_seq,mut_list[idx]))[-1]
    return y
def calc_phenotype(base_seq,mut_list):
    """
    Won't work for N mutations >16, for that change the below uint16 to be uint64 or something
    """
    N = len(mut_list)
    
    binary_array = np.unpackbits(np.arange(2**N, dtype=np.uint16).view(np.uint8)[:,None],axis=1)
    binary_array  = np.hstack((binary_array[1::2],binary_array[::2]))[:,-N:].astype(np.bool)
    y = np.zeros(2**N)
    for i,idx in enumerate(binary_array):
#         idx = np.array(list(base_str.format(i)),dtype=np.int).astype(np.bool)
        y[i] = RNA.fold(do_mut_list(base_seq,mut_list[idx]))[-1]
    return y
def GOP(pred,y):
    SST = np.sum(y**2)
    SSE = np.sum((pred-y)**2)
    return 1/(1+SSE/SST)

def get_order_list(n=10):
    binary_array = np.unpackbits(np.arange(2**n, dtype=np.uint16).view(np.uint8)[:,None],axis=1)
    binary_array  = np.hstack((binary_array[1::2],binary_array[::2]))[:,-n:]
    return np.sum(binary_array,axis=1)
def bootstrap_mean(arr,n_resamp = 10000,lower_per= 5,upper_per = 95):
    """
    bootstraps an estimate of the mean and the confidence interval
    """

    resampled_means = np.apply_along_axis(np.random.choice,axis=0,arr =arr,size=[n_resamp,arr.shape[0]],
                              replace=True).mean(axis=1)
    
    median = np.percentile(resampled_means,50,axis=0)
    ub = np.percentile(resampled_means,upper_per,axis=0)
    lb = np.percentile(resampled_means,lower_per,axis=0)
    yerr = np.vstack([np.abs(lb-median),ub-median])
    return median,lb,ub,yerr
BTS_to_Astrophysical_mappings = {
    'AGN': 'AGN',
    'AGN?': 'Anomaly',
    'CLAGN': 'AGN',
    'bogus?': 'Anomaly',
    'rock': 'Anomaly',
    'CV': 'CV',
    'CV?': 'CV',
    'AM CVn': 'CV',
    'varstar': 'Anomaly',
    "Cataclysmic": 'CV',
    'QSO': 'AGN', # AGN?
    'QSO?': 'AGN', # AGN?
    'NLS1': 'AGN', # AGN?
    'NLSy1?': 'AGN', # AGN?
    'Blazar': 'AGN', # AGN?
    'BL Lac': 'AGN', # AGN?
    'blazar': 'AGN', # AGN?
    'blazar?': 'AGN', # AGN?
    'Seyfert': 'AGN', # AGN?
    'star': 'Anomaly',
    'Ien': 'Anomaly',
    'LINER': 'Anomaly',
    'Ca-rich': 'Anomaly', 
    'FBOT': 'Anomaly',
    'ILRT': 'Anomaly',
    'LBV': 'Anomaly',
    'LRN': 'Anomaly',
    'SLSN-I': 'SLSN',
    'SLSN-I.5': 'SLSN',
    'SLSN-I?': 'SLSN',
    'SLSN-II': 'SLSN',
    'SN II': 'SN-II',
    'SN II-SL': 'SLSN',
    'SN II-norm': 'SN-II',
    'SN II-pec': 'Anomaly',
    'SN II?': 'SN-II',
    'SN IIL': 'SN-II',
    'SN IIP': 'SN-II',
    'SN IIb': 'Anomaly',
    'SN IIb-pec': 'Anomaly',
    'SN IIb?': 'Anomaly',
    'SN IIn': 'SN-II',
    'SN IIn?': 'SN-II',
    'SN Ia': 'SN-Ia',
    'SN Ia-00cx': 'Anomaly',# pec
    'SN Ia-03fg': 'Anomaly',# pec
    'SN Ia-91T': 'SN-Ia',
    'SN Ia-91bg': 'Anomaly',# pec
    'SN Ia-91bg?': 'Anomaly',# pec
    'SN Ia-99aa': 'Anomaly',
    'SN Ia-CSM': 'Anomaly',# pec
    'SN Ia-CSM?': 'Anomaly',# pec
    'SN Ia-norm': 'SN-Ia',
    'SN Ia-pec': 'Anomaly',# pec
    'SN Ia?': 'Anomaly',
    'SN Iax': 'Anomaly', # pec
    'SN Ib': 'SN-Ib/c',
    'SN Ib-pec': 'Anomaly',
    'SN Ib/c': 'SN-Ib/c',
    'SN Ib/c?': 'SN-Ib/c',
    'SN Ib?': 'SN-Ib/c',
    'SN Ibn': 'Anomaly',
    'SN Ibn?': 'Anomaly',
    'SN Ic': 'SN-Ib/c',
    'SN Ic-BL': 'Anomaly',
    'SN Ic-BL?': 'Anomaly',
    'SN Ic-SL': 'Anomaly',
    'SN Ic?': 'SN-Ib/c',
    'SN Icn': 'Anomaly',
    'TDE': 'Anomaly',
    'afterglow': 'Anomaly',
    'nova': 'CV',
    'nova-like': 'CV',
    'nova?': 'CV',
    "microlensing": 'Anomaly',
    "SN-Ib/c": "SN-Ib/c",

}

ZTF_sims_to_Astrophysical_mappings = {
    'SNIa-normal': 'SN-Ia',  
    'SNCC-II': 'SN-II',  
    'SNCC-Ibc': 'SN-Ib/c',   
    'SNCC-II': 'SN-II',    
    'SNCC-Ibc': 'SN-Ib/c',    
    'SNCC-II': 'SN-II',  
    'SNIa-91bg': 'SN-Ia',   
    'SNIa-x ': 'SN-Ia',  
    'KN': 'CV',  # TODO: Add CV class - no too worried about fine tuning
    'SLSN-I': 'SLSN-I',   
    'PISN': 'Anomaly',   
    'ILOT': 'Anomaly',    
    'CART': 'Anomaly',    
    'TDE': 'Anomaly',    
    'AGN': 'AGN',    
    'RRlyrae': 'Anomaly',   
    'Mdwarf': 'CV',    
    'EBE': 'Anomaly',    
    'MIRA': 'Anomaly',    
    'uLens-Binary': 'Anomaly',    
    'uLens-Point': 'Anomaly',    
    'uLens-STRING': 'Anomaly',    
    'uLens-Point': 'Anomaly',    
}

ztf_fid_to_filter = {
    1: 'g',
    2: 'r', 
    3: 'i' 
}

ztf_filter_to_fid = {
    'g': 1,
    'r': 2, 
    'i': 3, 
}

ztf_filters = ['g','r','i']
lsst_filters = ['u','g','r','i','z','Y']

# Order of images in the array
ztf_alert_image_order = ['science','reference','difference']
ztf_alert_image_dimension = (63, 63)

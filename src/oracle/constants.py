ELAsTiCC_to_Astrophysical_mappings = {
    'SNII-NMF': 'SNII', 
    'SNIc-Templates': 'SNIb/c', 
    'CART': 'CART', 
    'EB': 'EB', 
    'SNIc+HostXT_V19': 'SNIb/c', 
    'd-Sct': 'Delta Scuti', 
    'SNIb-Templates': 'SNIb/c', 
    'SNIIb+HostXT_V19': 'SNII', 
    'SNIcBL+HostXT_V19': 'SNIb/c', 
    'CLAGN': 'AGN', 
    'PISN': 'PISN', 
    'Cepheid': 'Cepheid', 
    'TDE': 'TDE', 
    'SNIa-91bg': 'SNI91bg', 
    'SLSN-I+host': 'SLSN', 
    'SNIIn-MOSFIT': 'SNII', 
    'SNII+HostXT_V19': 'SNII', 
    'SLSN-I_no_host': 'SLSN', 
    'SNII-Templates': 'SNII', 
    'SNIax': 'SNIax', 
    'SNIa-SALT3': 'SNIa', 
    'KN_K17': 'KN', 
    'SNIIn+HostXT_V19': 'SNII', 
    'dwarf-nova': 'Dwarf Novae', 
    'uLens-Binary': 'uLens', 
    'RRL': 'RR Lyrae', 
    'Mdwarf-flare': 'M-dwarf Flare', 
    'ILOT': 'ILOT', 
    'KN_B19': 'KN', 
    'uLens-Single-GenLens': 'uLens', 
    'SNIb+HostXT_V19': 'SNIb/c', 
    'uLens-Single_PyLIMA': 'uLens'
}

BTS_to_Astrophysical_mappings = {

    'Ca-rich': 'CART', 
    'FBOT': 'FBOT',
    'ILRT': 'ILRT',
    'LBV': 'LBV',
    'LRN': 'LRN',
    'SLSN-I': 'SLSN-I',
    'SLSN-I.5': 'SLSN-I',
    'SLSN-I?': 'SLSN-I',
    'SLSN-II': 'SLSN-II',
    'SN II': 'SN-II-normal',
    'SN II-SL': 'SLSN-II',
    'SN II-norm': 'SN-II-normal',
    'SN II-pec': 'SN-II-peculiar',
    'SN II?': 'SN-II-normal',
    'SN IIL': 'SN-II-normal',
    'SN IIP': 'SN-II-normal',
    'SN IIb': 'SN-II-peculiar',
    'SN IIb-pec': 'SN-IIb',
    'SN IIb?': 'SN-IIb',
    'SN IIn': 'SN-IIn',
    'SN IIn?': 'SN-IIn',
    'SN Ia': 'SN-Ia-normal',
    'SN Ia-00cx': 'SN-Ia-peculiar',# pec
    'SN Ia-03fg': 'SN-Ia-peculiar',# pec
    'SN Ia-91T': 'SN-Ia-normal',
    'SN Ia-91bg': 'SN-Ia-peculiar',# pec
    'SN Ia-91bg?': 'SN-Ia-peculiar',# pec
    'SN Ia-99aa': 'SN-Ia-normal',
    'SN Ia-CSM': 'SN-Ia-peculiar',# pec
    'SN Ia-CSM?': 'SN-Ia-peculiar',# pec
    'SN Ia-norm': 'SN-Ia-normal',
    'SN Ia-pec': 'SN-Ia-peculiar',# pec
    'SN Ia?': 'SN-Ia',
    'SN Iax': 'SN-Ia-peculiar', # pec
    'SN Ib': 'SN-Ib/c',
    'SN Ib-pec': 'SN-Ib-peculiar',
    'SN Ib/c': 'SN-Ib/c',
    'SN Ib/c?': 'SN-Ib/c',
    'SN Ib?': 'SN-Ib/c',
    'SN Ibn': 'SN-Ibn',
    'SN Ibn?': 'SN-Ibn',
    'SN Ic': 'SN-Ib/c-normal',
    'SN Ic-BL': 'SN-Ic-BL',
    'SN Ic-BL?': 'SN-Ic-BL',
    'SN Ic-SL': 'SN-Ic-SL',
    'SN Ic?': 'SN-Ib/c',
    'SN Icn': 'SN-Icn',
    'TDE': 'TDE',
    'afterglow': 'afterglow',
    'nova': 'nova/nova-like',
    'nova-like': 'nova/nova-like',
    'nova?': 'nova/nova-like',

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

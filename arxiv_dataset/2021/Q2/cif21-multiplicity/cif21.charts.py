# %%

# LATEX CHART GENERATOR
# Preliminar chart for describing multiple systems in .tex format
# Cifuentes et al. (2021)

import numpy as np
import csv
import pandas as pd
import textwrap

filename = 'cif21.multiplicity'
df = pd.read_csv(filename+'.csv')

for i in range(len(df)-1):
    if df['ID_system'][i] == df['ID_system'][i+1]:
        Name = str(df['Name'][i])
        if pd.notnull(df['Koenigstuhl'][i]) == False:
            KO_name = ''
            KO = ''
        else:
            KO_name = ', K\\"onigstuhl '
            KO = str(df['Koenigstuhl'][i])
        if pd.notnull(df['Karmn'][i]) == False:
            Karmn_name = ''
            Karmn = ''
        else:
            Karmn_name = ', Karmn '
            Karmn = str(df['Karmn'][i])
        SpT = str(df['SpT'][i])
        Gaia_id = str(df['gaia_id'][i])
        RA = str(df['RA_J2016'][i])
        DE = str(df['DE_J2016'][i])
        Plx = format(df['parallax'][i], '.2f')
        ePlx = format(df['parallax_error'][i], '.2f')
        d = format(1000/df['parallax'][i], '.2f')
        ed = format((1000*df['parallax_error'][i]/df['parallax'][i]**2), '.2f')
        pmra = format(df['pmra'][i], '.3f')
        pmra_error = format(df['pmra_error'][i], '.3f')
        pmdec = format(df['pmdec'][i], '.3f')
        pmdec_error = format(df['pmdec_error'][i], '.3f')
        mu_ratio = format(df['muratio_AB'][i], '.3f')
        delta_PA = format(df['deltaPA_AB'][i], '.3f')
        rho = format(df['rho_AB'][i], '.2f')
        delta_d = format(
            np.abs((df['d_B'][i] - df['d_A'][i])/df['d_A'][i]), '.3f')
        s = format(1000/df['parallax'][i] * df['rho_AB'][i], '.0f')
        theta = format(df['theta_AB'][i], '.1f')
        RUWE = format(df['ruwe'][i], '.3f')
        Vr = format(df['radial_velocity'][i], '.3f')
        eVr = format(df['radial_velocity_error'][i], '.3f')
        Lbol = format(1E4*df['Lbol'][i], '.2f')
        eLbol = format(1E4*df['Lberr'][i], '.2f')
        Teff = format(df['Teff'][i], '.0f')
        eTeff = format(df['e_Teff'][i], '.0f')
        Mass = format(df['Mass_A'][i], '.4f')
        Porb = format(df['Porb_AB'][i]/1E3, '.1f')
        Ug = format(df['Ug_AB'][i], '.1f')
        G_mag = format(df['phot_g_mean_mag'][i], '.4f')
        eG_mag = format(df['phot_g_mean_mag_error'][i], '.4f')
        J_mag = format(df['Jmag'][i], '.3f')
        eJ_mag = format(df['e_Jmag'][i], '.3f')
        Qf_2M = str(df['Qfl'][i])
        Qf_Ws = str(df['qph'][i])
        if df['ID_system'][i+1] == df['ID_system'][i]:
            WDS_name_B = str(df['WDS_name'][i+1])
            WDS_disc_B = str(df['WDS_disc'][i+1])
            Name_B = str(df['Name'][i+1])
            if pd.notnull(df['Koenigstuhl'][i+1]) == False:
                KO_B_name = ''
                KO_B = ''
            else:
                KO_B_name = ', K\\"onigstuhl '
                KO_B = str(df['Koenigstuhl'][i+1])
            if pd.notnull(df['Karmn'][i+1]) == False:
                Karmn_B_name = ''
                Karmn_B = ''
            else:
                Karmn_B_name = ', Karmn '
                Karmn_B = str(df['Karmn'][i+1])
            Discoverer_B = str(df['Discoverer'][i+1])
            SpT_B = str(df['SpT'][i+1])
            Gaia_id_B = str(df['gaia_id'][i+1])
            RA_B = str(df['RA_J2016'][i+1])
            DE_B = str(df['DE_J2016'][i+1])
            ra_B = str(df['ra'][i+1])
            dec_B = str(df['dec'][i+1])
            Plx_B = format(df['parallax'][i+1], '.2f')
            ePlx_B = format(df['parallax_error'][i+1], '.2f')
            d_B = format(1000/df['parallax'][i+1], '.2f')
            ed_B = format((1000*df['parallax_error'][i+1] /
                           df['parallax'][i+1]**2), '.2f')
            pmra_B = format(df['pmra'][i+1], '.3f')
            pmra_error_B = format(df['pmra_error'][i+1], '.3f')
            pmdec_B = format(df['pmdec'][i+1], '.3f')
            pmdec_error_B = format(df['pmdec_error'][i+1], '.3f')
            RUWE_B = format(df['ruwe'][i+1], '.3f')
            Vr_B = format(df['radial_velocity'][i+1], '.3f')
            eVr_B = format(df['radial_velocity_error'][i+1], '.3f')
            Lbol_B = format(1E4*df['Lbol'][i+1], '.2f')
            eLbol_B = format(1E4*df['Lberr'][i+1], '.2f')
            Teff_B = format(df['Teff'][i+1], '.0f')
            eTeff_B = format(df['e_Teff'][i+1], '.0f')
            Mass_B = format(df['Mass_B'][i], '.4f')
            G_mag_B = format(df['phot_g_mean_mag'][i+1], '.4f')
            eG_mag_B = format(df['phot_g_mean_mag_error'][i+1], '.4f')
            J_mag_B = format(df['Jmag'][i+1], '.3f')
            eJ_mag_B = format(df['e_Jmag'][i+1], '.3f')
            Qf_2M_B = str(df['Qfl'][i+1])
            Qf_Ws_B = str(df['qph'][i+1])

    X = '\ldots'

    file = open('Chart_'+str(i)+'.tex', 'w')

    text = """
\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\usepackage[sc]{mathpazo} % Use the Palatino font

\\linespread{1.05} % Line spacing - Palatino needs more space between lines
\\usepackage{microtype} % Slightly tweak font spacing for aesthetics
\\usepackage[top=32mm,bottom=15mm,left=10mm,right=10mm]{geometry}
\\usepackage{titling}
\\usepackage{tabularx}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{color}
\\usepackage{floatrow}
\\pagenumbering{gobble}

\\newfloatcommand{capbtabbox}{table}[][\\FBwidth]

\\definecolor{OK}{RGB}{0, 180, 0}
\\definecolor{KO}{RGB}{230, 0, 15}
\\definecolor{MEH}{RGB}{225, 128, 0}

\\setlength{\\droptitle}{-8\\baselineskip} % Move the title up
\\title{\\Huge \\textcolor{black}{WDS """ + WDS_name_B.replace('nan', '\\ldots') + """ (""" + WDS_disc_B.replace('nan', '\\ldots') + """)}}
\\author{
{\\Large {\\bf A:} """ + Name + KO_name + KO + Karmn_name + Karmn + """}\\vspace{.5cm}\\\\
{\\Large {\\bf B:} """ + Name_B + KO_B_name + KO_B + Karmn_B_name + Karmn_B + """}
}
\\date{}

\\begin{document}

\\maketitle

\\begin{figure}[!h]
\\begin{floatrow}
\\ffigbox{
    \\includegraphics[width=0.4\\paperwidth]{Figures/""" + Name.replace(' ', '_').replace('.', '_') + """.png}
    }{}
\\capbtabbox{%
    {
    \\begin{tabular}[]{r|c|l} 
    %\\noalign{\\smallskip}
    $\\rho$ & """ + rho.replace('nan', '\\ldots') + """ & arcsec \\\\
    $\\theta$ & """ + theta.replace('nan', '\\ldots') + """ & deg \\\\
    $\\mu$ ratio & \\textcolor{OK}{""" + mu_ratio.replace('nan', '\\ldots') + """} &  \\\\
    $\\Delta PA$ & \\textcolor{OK}{""" + delta_PA.replace('nan', '\\ldots') + """} & deg \\\\
    $\\Delta d/d$ & \\textcolor{OK}{""" + delta_d.replace('nan', '\\ldots') + """} &  \\\\
    $d$ & """ + d.replace('nan', '\\ldots') + """ & pc \\\\
    $s$ & """ + s.replace('nan', '\\ldots') + """ & au \\\\
    $P_{\\rm orb}$ & """ + Porb.replace('nan', '\\ldots') + """ & 10$^{3}$ a \\\\
    $-U_g^*$ & """ + Ug.replace('nan', '\\ldots') + """ & 10$^{33}$ J \\\\
    %\\noalign{\\smallskip}
    \\end{tabular}
    }
}{}
\\end{floatrow}
\\end{figure}

\\begin{table}[!h]
    \\begin{tabular}{r|cc|l}
        \\centering
        Component & {\\bf A} & {\\bf B} & \\\\
        SpT & """ + SpT + """ & """ + SpT_B + """ &\\\\
        $\\alpha$  & """ + RA.replace('-', '--') + """  & """ + RA_B.replace('-', '--') + """ & \\\\
        $\\delta$ & """ + DE.replace('-', '--') + """  & """ + DE_B.replace('-', '--') + """ & \\\\
        $\\pi$ & """ + Plx + """ $\\pm$ """ + ePlx + """ & """ + Plx_B + """ $\\pm$ """ + ePlx_B + """ & mas \\\\
        $\\mu_{\\alpha}\\cos{\\delta}$ & $""" + pmra + """$ $\\pm$ $""" + pmra_error + """$ & $""" + pmra_B + """$ $\\pm$ $""" + pmra_error_B + """$ & mas a$^{-1}$ \\\\
        $\\mu_{\\delta}$  & $""" + pmdec + """$ $\\pm$ $""" + pmdec_error + """$ & $""" + pmdec_B + """$ $\\pm$ $""" + pmdec_error_B + """$ & mas a$^{-1}$ \\\\        
        $\\gamma$ & $""" + Vr.replace('nan', '\\ldots') + """$ $\\pm$ $""" + eVr.replace('nan', '\\ldots') + """$ & $""" + Vr_B.replace('nan', '\\ldots') + """$ $\\pm$ $""" + eVr_B.replace('nan', '\\ldots') + """$ & km s$^{-1}$ \\\\
        %
        $G$ & """ + G_mag + """ $\\pm$ """ + eG_mag + """ & """ + G_mag_B + """ $\\pm$ """ + eG_mag_B + """ & mag \\\\
        $J$ & """ + J_mag + """ $\\pm$ """ + eJ_mag + """ & """ + J_mag_B.replace('nan', '\\ldots') + """ $\\pm$ """ + eJ_mag_B.replace('nan', '\\ldots') + """ & mag \\\\
        %
        $L$ & """ + Lbol + """ $\\pm$ """ + eLbol + """ & """ + Lbol_B + """ $\\pm$ """ + eLbol_B + """ & 10$^{-4} L_\\odot$ \\\\
        $T_{\\rm eff}$ & """ + Teff + """ $\\pm$ """ + eTeff + """ &  """ + Teff_B + """ $\\pm$ """ + eTeff_B + """ & K \\\\
        $\\mathcal{M}$ & """ + Mass.replace('nan', '\\ldots') + """ & """ + Mass_B.replace('nan', '\\ldots') + """ & $\\mathcal{M_\\odot}$ \\\\
        %
        {\\tt RUWE} & \\textcolor{OK}{""" + RUWE.replace('nan', '\\ldots') + """} & \\textcolor{OK}{""" + RUWE_B.replace('nan', '\\ldots') + """} & \\\\
        Qflag 2MASS & """ + Qf_2M + """ & """ + Qf_2M_B.replace('nan', '\\ldots') + """ & \\\\
        Qflag AllWISE & """ + Qf_Ws + """ & """ + Qf_Ws_B.replace('nan', '\\ldots') + """ & \\\\    
    \\end{tabular}
\\end{table}

\\large{\\noindent Comments go here. Comments go here. Comments go here. Comments go here. Comments go here. Comments go here. Comments go here. Comments go here. Comments go here. Comments go here. Comments go here.}

\\end{document}

"""
    file.write(text.replace('nan $\\pm$ nan', '\\ldots').replace(
        '$\ldots$ $\pm$ $\ldots$', '\\ldots'))

file.close()

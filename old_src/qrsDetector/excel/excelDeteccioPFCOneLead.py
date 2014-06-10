
# -*- coding: utf8 -*-

from tempfile import TemporaryFile
from xlwt import Workbook, easyxf, BIFFRecords
from system.settings import *

def getReqInfo(path_name, lead):

    from os import listdir
    from error.stats import compareLists
    #files = [x for x in listdir('resultats/resultatsDeteccio/secondRound/'+path_name) if x.endswith("_"+str(lead))]
    cases = {}
    resum = []
    import cPickle
    contador = 0
    cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
    f = open('resultats_mix_detection.txt','rb')
    ZZ = cPickle.load(f)

    for i,case in enumerate(cases):
        try:
            f.close()
            
            Z = ZZ[case]
            #labels = h_io.read_labels(case, REDIAGNOSE_DIAGNOSER)
            #real_pos = [pos for ind_lab, pos in enumerate(labels[0]) if labels[1][i] in 'SVN']
            
            #tp, fp, fn, sensibility, vpp = compareLists(real_pos, Z)
            tp,fp,fn,sensibility,vpp = Z
            #res cntiene  tp, fp, fn, float(tp)/(tp+fn), float(tp)/(tp+fp)
            if tp == 0:
                continue
            cases[contador] = case
            resum.append([tp+fn,tp, fp, fn, sensibility, vpp])
            contador = contador + 1
        except:
            continue
    return resum, cases

def afegeixAlResum(sheet,lead_name, resum, explanation,vert_ref):
    
    sheet.portrait = True
    sheet.left_margin = 0.25
    sheet.right_margin = 0.25
    sheet.top_margin = 0.25
    sheet.bottom_margin = 0.25
    
    
    st_value_letra_gorda = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 130;');
    st_value = easyxf('align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 130;');
    st_value_right = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    st_value_down = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    st_value_corner = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    st_value_senspec = easyxf('pattern: pattern solid, fore_colour pale_blue;''align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    
    style_title = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 150;')
    style_case = easyxf('pattern: pattern fine_dots, fore_colour gray25;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 140;')
    style_case2 = easyxf('pattern: pattern fine_dots, fore_colour gray50;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 140;')
    
    style2_title_vert = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''alignment: rotation 90;''borders: left hair, right hair, top hair, bottom hair;''font: height 150;')
    style_none = easyxf('align: vertical center, horizontal center;''borders: left  no_line, right  no_line, top  no_line, bottom  no_line;''font: height 150;')
    
    
         
    
    from numpy import mod   
    if mod(len(resum),2)==0:
        zz = len(resum)/2
    else:
        zz = len(resum)/2+1

    from numpy import array
    rr = sum(array(resum),0)
    sheet.write(vert_ref,0,lead_name,style = style_case2)
    sheet.write(vert_ref,1,str(rr[0]),style = style_case)

    sheet.write(vert_ref,2,str(rr[1]),style = st_value)
    sheet.write(vert_ref,3,str(rr[2]),style = st_value)
    sheet.write(vert_ref,4,str(rr[3]),style = st_value)

    from numpy import mean
    rr = mean(array(resum),0)
    sheet.write(vert_ref,5,str(round(rr[4]*100,3)),style = st_value)
    sheet.write(vert_ref,6,str(round(rr[5]*100,3)),style = st_value)


def afegeixFull(book, lead_name,resum,cases,descripcio):
    sheet = book.add_sheet('Resultats deteccio'+lead_name)
    sheet.portrait = True
    sheet.left_margin = 0.25
    sheet.right_margin = 0.25
    sheet.top_margin = 0.25
    sheet.bottom_margin = 0.25
    
    st_value_letra_gorda = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 130;');
    st_value = easyxf('align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 130;');
    st_value_right = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    st_value_down = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    st_value_corner = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    st_value_senspec = easyxf('pattern: pattern solid, fore_colour pale_blue;''align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
    
    style_title = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 150;')
    style_case = easyxf('pattern: pattern fine_dots, fore_colour gray25;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 140;')
    style_case2 = easyxf('pattern: pattern fine_dots, fore_colour gray50;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 140;')
    
    style2_title_vert = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''alignment: rotation 90;''borders: left hair, right hair, top hair, bottom hair;''font: height 150;')
    style_none = easyxf('align: vertical center, horizontal center;''borders: left  no_line, right  no_line, top  no_line, bottom  no_line;''font: height 150;')
    
    
    def plot_columna_resum(ref_ini, resum, cases, sheet,zz):
        sheet.write(1,ref_ini+0,'   Caso   ',style = style_title)
        sheet.write(1,ref_ini+1,'Total Beats',style = style_title)
        sheet.write(1,ref_ini+2,'    TP     ',style = style_title)
        sheet.write(1,ref_ini+3,'    FP     ',style = style_title)
        sheet.write(1,ref_ini+4,'    FN     ',style = style_title)
        sheet.write(1,ref_ini+5,'Sensibility',style = style_title)
        sheet.write(1,ref_ini+6,'    PPV    ',style = style_title)
        
        sheet.col(ref_ini+0).width = 2000
        sheet.col(ref_ini+1).width = 2000
        sheet.col(ref_ini+2).width = 2000
        sheet.col(ref_ini+3).width = 2000
        sheet.col(ref_ini+4).width = 2000
        sheet.col(ref_ini+5).width = 2000
        sheet.col(ref_ini+6).width = 2000
        
        for i,x in enumerate(resum):
            sheet.write(2+i,ref_ini+0,cases[i+zz],style = style_case2)
            sheet.write(2+i,ref_ini+1,resum[i][0],style=style_case)
            sheet.write(2+i,ref_ini+2,resum[i][1],style=style_case)
            sheet.write(2+i,ref_ini+3,resum[i][2],style=style_case)
            sheet.write(2+i,ref_ini+4,resum[i][3],style=st_value)
            
            from numpy import round
            sheet.write(2+i,ref_ini+5,round(resum[i][4]*100,3),style=st_value)
            sheet.write(2+i,ref_ini+6,round(resum[i][5]*100,3),style=st_value)
            
        return ref_ini+7
         
    
    from numpy import mod   
    if mod(len(resum),2)==0:
        zz = len(resum)/2
    else:
        zz = len(resum)/2+1
    
    sheet.write_merge(0,0,0,13,"Resultados de la deteccion " + descripcio + " en la derivacion " + lead_name,style = style_title)
    ax = plot_columna_resum(0, resum[:zz], cases, sheet,0)
    ax = plot_columna_resum(ax, resum[zz:],cases, sheet,zz)


def creaLibro(actual_path, nom_test, explanation = '', leads = [0,1,2],lead_name = ['C1','C2','C3']):
    book = Workbook()
    
    style_case2 = easyxf('pattern: pattern fine_dots, fore_colour gray50;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 140;')
    style_title = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 150;')
    
    
    full_resum = book.add_sheet('Resum resultats')
    full_resum.write_merge(0,1,0,6,"Resultados de la deteccion\n" + explanation,style = style_title)
    full_resum.write(2,0,'Lead',style = style_case2)
    full_resum.write(2,1,'Total\nBeats',style = style_case2)

    full_resum.write(2,2,'TP',style = style_case2)
    full_resum.write(2,3,'FP',style = style_case2)
    full_resum.write(2,4,'FN',style = style_case2)
    full_resum.write(2,5,'Sensibility',style = style_case2)
    full_resum.write(2,6,'    PPV    ',style = style_case2)

    full_resum.col(0).width = 2000
    full_resum.col(1).width = 2000
    full_resum.col(2).width = 2000
    full_resum.col(3).width = 2000
    full_resum.col(4).width = 2000
    full_resum.col(5).width = 2000
    full_resum.col(6).width = 2000


    resum, cases = getReqInfo(actual_path,'')
    afegeixFull(book,"unified leads", resum,cases, explanation)
    afegeixAlResum(full_resum,"unified leads", resum, explanation,3)


    book.save(nom_test)
    book.save(TemporaryFile())


cont = 0

nom_test = 'deteccion PanTomkins umbral adaptativo joined leads.xls'
creaLibro(" ",nom_test,"deteccion PanTomkins usando umbral adaptativo joined leads")
print str("done joined leads")

'''
try:
    actual_path = 'pantomkinsAdapt'
    nom_test = 'deteccion PanTomkins umbral adaptativo.xls'
    creaLibro(actual_path,nom_test," deteccion PanTomkins usando umbral adaptativo ")
    print str("done " + actual_path)
    cont = cont + 1
except:
    print str(cont)

'''

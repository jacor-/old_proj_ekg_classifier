
# -*- coding: utf8 -*-

from tempfile import TemporaryFile
from xlwt import Workbook, easyxf, BIFFRecords
from system.settings import *


def getCasos(base_filename, path):
    cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
    a = []
    b = []
    
    Xs_C1 = []
    Xs_C2 = []
    Xs_C3 = []
    Xs_Combined = []
    
    Ys = []
    indexs = [0]
    case_index = {}
    
    for num_case, case in enumerate(cases[:]):
        try:
            print case
            import cPickle
            #f = open('clasificacionPFC/'+case+'_lead1_4_components','rb')
            #Ys = Ys + list(cPickle.load(f))
            #f.close()
            
            
            #f = open(path+case+'_lead1_4_'+base_filename,'rb')
            #lead1 = cPickle.load(f)
            #f.close()
            #Xs_C1 = Xs_C1 + list(lead1)
        
            
            #f = open(path+case+'_lead2_4_'+base_filename,'rb')
            #lead2 = cPickle.load(f)
            #f.close()
            #Xs_C2 = Xs_C2 + list(lead2)
        
            
            #f = open(path+case+'_lead3_4_'+base_filename,'rb')
            #lead3 = cPickle.load(f)
            #f.close()
            #Xs_C3 = Xs_C3 + list(lead3)
            
            #f = open(path+case+'_lead_all_4_components','rb')
            #lead_tot = cPickle.load(f)
            #f.close()
            #Xs_Combined = Xs_Combined + list(lead_tot)
            
            f = open(path+case+'_valid_labs_4_components','rb')
            #f = open(path+case+'_valid_labs_4_componentesPruebaJoseDef','rb')
            
            valid_labs = cPickle.load(f)
            f.close()
            tr = {'V':1,'N':0,'S':0,0:0,1:1} 
            '''
            CONFLICTO POTENCIAL! OJO AL valid_labs! A vegades es etiquetes i posicions i a vegades nomes te etiquetes.
            '''   
            Z = map(lambda x:tr[x],valid_labs)
            Ys = Ys + list(Z)
            
            indexs.append(len(Ys))
            case_index[num_case] = case
            
        except:
            print "   error " + case
            pass
    return case_index,Ys,indexs

def getReqInfo(actual_file):
    import cPickle
    f=open(actual_file,'rb')
    Z = cPickle.load(f)
    del Z
    res = cPickle.load(f)
    
    path = 'clasificacionPFC/'
    base_filename = 'components_no_norm'
    base_filename = 'mahalanobius_no_normalized'

    case_index,Ys,indexs = getCasos(base_filename,path)
    return res, case_index

def getReqInfo22(actual_file):
    import cPickle
    f=open(actual_file,'rb')
    Z = cPickle.load(f)
    res = cPickle.load(f)
    
    path = 'clasificacionPFC/'
    base_filename = 'components_no_norm'
    base_filename = 'mahalanobius_no_normalized'

    case_index,Ys,indexs = getCasos(base_filename,path)
    return Z, case_index, Ys, indexs



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
    sheet.write(vert_ref,0,lead_name,style = style_case)
    sheet.write(vert_ref,1,str(rr[0]),style = style_case)
    
    sheet.write(vert_ref,2,str(rr[2]),style = style_case)
    sheet.write(vert_ref,3,str(rr[1]),style = style_case)
    
    from numpy import round
    
    
    sheet.write(vert_ref,4,rr[4],style = st_value)
    sheet.write(vert_ref,5,rr[3],style = st_value)
    sheet.write(vert_ref,6,rr[6],style = st_value)
    sheet.write(vert_ref,7,rr[5],style = st_value)
        
    z = float(rr[4])/(rr[4]+rr[5])
    z = str(round(z*100,3))
    sheet.write(vert_ref,8,z,style = st_value)

    z = float(rr[3])/(rr[3]+rr[6])
    z = str(round(z*100,3))
    sheet.write(vert_ref,9,z,style = st_value)

    z = float(rr[4])/(rr[4]+rr[6])
    z = str(round(z*100,3))
    sheet.write(vert_ref,10,z,style = st_value)
    
    z = float(rr[3])/(rr[3]+rr[5])
    z = str(round(z*100,3))
    sheet.write(vert_ref,11,z,style = st_value)
    
    
    



def afegeixFull(book, lead_name,resum,cases,descripcio):
    sheet = book.add_sheet('Resultats '+lead_name)
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
        sheet.write_merge(1,2,ref_ini+0,ref_ini+0,'Caso',style = style_title)
        sheet.write_merge(1,2,ref_ini+1,ref_ini+1,'Total\nBeats',style = style_title)
        sheet.write_merge(1,2,ref_ini+2,ref_ini+2,'Normal\nbeats',style = style_title)
        sheet.write_merge(1,2,ref_ini+3,ref_ini+3,'VPC\nbeats',style = style_title)
        
        sheet.write(1,ref_ini+4,'TP: N',style = style_title)
        sheet.write(2,ref_ini+4,'TN: V',style = style_title)
        
        sheet.write(1,ref_ini+5,'TN: N',style = style_title)
        sheet.write(2,ref_ini+5,'TP: V',style = style_title)
        
        sheet.write(1,ref_ini+6,'FP: N',style = style_title)
        sheet.write(2,ref_ini+6,'FN: V',style = style_title)
        
        sheet.write(1,ref_ini+7,'FN: N',style = style_title)
        sheet.write(2,ref_ini+7,'FP: V',style = style_title)
        
        
        
        sheet.col(ref_ini+0).width = 1700
        sheet.col(ref_ini+1).width = 1500
        sheet.col(ref_ini+2).width = 1500
        sheet.col(ref_ini+3).width = 1500
        sheet.col(ref_ini+4).width = 1200
        sheet.col(ref_ini+5).width = 1200
        sheet.col(ref_ini+6).width = 1200
        sheet.col(ref_ini+7).width = 1200
        
        for i,x in enumerate(resum):
            sheet.write(3+i,ref_ini+0,cases[i+zz],style = style_case2)
            sheet.write(3+i,ref_ini+1,resum[i][0],style=style_case)
            sheet.write(3+i,ref_ini+2,resum[i][2],style=style_case)
            sheet.write(3+i,ref_ini+3,resum[i][1],style=style_case)
            sheet.write(3+i,ref_ini+4,resum[i][4],style=st_value)
            sheet.write(3+i,ref_ini+5,resum[i][3],style=st_value)
            sheet.write(3+i,ref_ini+6,resum[i][6],style=st_value)
            sheet.write(3+i,ref_ini+7,resum[i][5],style=st_value)
        return ref_ini+8
         
    
    from numpy import mod   
    if mod(len(resum),2)==0:
        zz = len(resum)/2
    else:
        zz = len(resum)/2+1
    
    sheet.write_merge(0,0,0,16,"Resultados de la clasificacion " + descripcio + " en la derivacion " + lead_name,style = style_title)
    ax = plot_columna_resum(0, resum[:zz], cases, sheet,0)
    ax = plot_columna_resum(ax, resum[zz:],cases, sheet,zz)


def creaLibro(actual_file, nom_test, explanation = '',lead_name = ['C1','C2','C3']):
    book = Workbook()
    
    style_case2 = easyxf('pattern: pattern fine_dots, fore_colour gray50;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 140;')
    style_title = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 150;')
    full_resum = book.add_sheet('Resum resultats')
    full_resum.write_merge(0,1,0,11,"Resultados de la clasificacion\n" + explanation,style = style_title)
    full_resum.write_merge(2,3,0,0,'Lead',style = style_case2)
    full_resum.write_merge(2,3,1,1,'Total\nBeats',style = style_case2)
    full_resum.write_merge(2,3,2,2,'Normal\nbeats',style = style_case2)
    full_resum.write_merge(2,3,3,3,'VPC\nbeats',style = style_case2)

    full_resum.write(2,4,'TP: N',style = style_case2)
    full_resum.write(3,4,'TN: V',style = style_case2)
    
    full_resum.write(2,5,'TN: N',style = style_case2)
    full_resum.write(3,5,'TP: V',style = style_case2)
    
    full_resum.write(2,6,'FP: N',style = style_case2)
    full_resum.write(3,6,'FN: V',style = style_case2)
    
    full_resum.write(2,7,'FN: N',style = style_case2)
    full_resum.write(3,7,'FP: V',style = style_case2)

    
    full_resum.write(2,8,'Sensibility: N',style = style_case2)
    full_resum.write(2,9,'Specificity: N',style = style_case2)
    full_resum.write(2,10,'PPV: N',style = style_case2)
    full_resum.write(2,11,'NPV: N',style = style_case2)

    full_resum.write(3,8,'Specificity: V',style = style_case2)
    full_resum.write(3,9,'Sensibility: V',style = style_case2)
    full_resum.write(3,10,'NPV: V',style = style_case2)
    full_resum.write(3,11,'PPV: V',style = style_case2)


    full_resum.col(0).width = 2000
    full_resum.col(1).width = 2000
    full_resum.col(2).width = 2000
    full_resum.col(3).width = 2000
    full_resum.col(4).width = 2000
    full_resum.col(5).width = 2000
    full_resum.col(6).width = 2000
    full_resum.col(7).width = 2000


    for i,file in enumerate(actual_file):
        resum, cases = getReqInfo(file)
        afegeixFull(book,lead_name[i], resum,cases, explanation)
        afegeixAlResum(full_resum,lead_name[i], resum, explanation,4+i)
    
     
    book.save(nom_test)
    book.save(TemporaryFile())

def creaOrLogic1(actual_file, nom_test, explanation = '',lead_name = ['C1','C2','C3']):
    book = Workbook()
    style_case2 = easyxf('pattern: pattern fine_dots, fore_colour gray50;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 140;')
    style_title = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''borders: left 7, right 7, top 7, bottom 7;''font: height 150;')
    full_resum = book.add_sheet('Resum resultats')
    full_resum.write_merge(0,1,0,11,"Resultados de la clasificacion uniendo derivaciones\n" + explanation,style = style_title)
    full_resum.write_merge(2,3,0,0,'Lead',style = style_case2)
    full_resum.write_merge(2,3,1,1,'Total\nBeats',style = style_case2)
    full_resum.write_merge(2,3,2,2,'Normal\nbeats',style = style_case2)
    full_resum.write_merge(2,3,3,3,'VPC\nbeats',style = style_case2)
    full_resum.write(2,4,'TP: N',style = style_case2)
    full_resum.write(3,4,'TN: V',style = style_case2)
    
    full_resum.write(2,5,'TN: N',style = style_case2)
    full_resum.write(3,5,'TP: V',style = style_case2)
    
    full_resum.write(2,6,'FP: N',style = style_case2)
    full_resum.write(3,6,'FN: V',style = style_case2)
    
    full_resum.write(2,7,'FN: N',style = style_case2)
    full_resum.write(3,7,'FP: V',style = style_case2)

    
    full_resum.write(2,8,'Sensibility: N',style = style_case2)
    full_resum.write(2,9,'Specificity: N',style = style_case2)
    full_resum.write(2,10,'PPV: N',style = style_case2)
    full_resum.write(2,11,'NPV: N',style = style_case2)

    full_resum.write(3,8,'Specificity: V',style = style_case2)
    full_resum.write(3,9,'Sensibility: V',style = style_case2)
    full_resum.write(3,10,'NPV: V',style = style_case2)
    full_resum.write(3,11,'PPV: V',style = style_case2)


    full_resum.col(0).width = 2000
    full_resum.col(1).width = 2000
    full_resum.col(2).width = 2000
    full_resum.col(3).width = 2000
    full_resum.col(4).width = 2000
    full_resum.col(5).width = 2000
    full_resum.col(6).width = 2000
    full_resum.col(7).width = 2000


    resum_global = []
    for i,file in enumerate(actual_file):
        resum, cases,Ys,indexs = getReqInfo22(file)
        resum_global.append(resum)
    #return resum_global
    from numpy import array,sum
    
    
    
    def getMatrixResult(indexs,Z_ref,Z_res):
        res = []
        for i in range(len(indexs)-1):
            Z_ref_loc = Z_ref[indexs[i]:indexs[i+1]]
            Z_res_loc = Z_res[indexs[i]:indexs[i+1]]
            FP = sum(Z_res_loc-Z_ref_loc==1)
            FN = sum(Z_res_loc-Z_ref_loc==-1)
            res.append([len(Z_ref_loc),sum(Z_ref_loc),len(Z_ref_loc)-sum(Z_ref_loc),sum(Z_ref_loc)-FN,len(Z_ref_loc)-sum(Z_ref_loc)-FP,FP,FN])
        return res


    resum_global = sum(array(resum_global),0)    
    resum_global2 = (resum_global >= 1).astype('int')
    resum_global2 = getMatrixResult(indexs,Ys,resum_global2)
    afegeixFull(book," 1 o mes", resum_global2,cases, "resum " + explanation)
    afegeixAlResum(full_resum," 1 o mes", resum_global2, "resum " + explanation,4)
    
    resum_global2 = (resum_global >= 2).astype('int')
    resum_global2 = getMatrixResult(indexs,Ys,resum_global2)    
    afegeixFull(book," 2 o mes", resum_global2,cases, "resum " + explanation)
    afegeixAlResum(full_resum," 2 o mes", resum_global2, "resum " + explanation,5)
    
    resum_global2 = (resum_global >= 3).astype('int')
    resum_global2 = getMatrixResult(indexs,Ys,resum_global2)
    afegeixFull(book," 3 o mes", resum_global2,cases, "resum " + explanation)
    afegeixAlResum(full_resum," 3 o mes", resum_global2, "resum " + explanation,6)
    
    book.save('join-leads'+nom_test)
    book.save(TemporaryFile())










'''
## mohaeavioiaoaus reduced
try:
    print str("aaa")
    actual_file = ['predict_caracterizacionASacoCovarianzaTotalLocalYCovarianzaNormalGeneral','predict_caracterizacionASacoCovarianzaTotalLocalYCovarianzaNormalGeneral','predict_caracterizacionASacoCovarianzaTotalLocalYCovarianzaNormalGeneral']
    print str("bbb")
    nom_test = 'laSegundaDerivacionEsCaracterizacionConTodoMohoIntraYEntercdSinReducirNa.xls'
    print str("ccc")
    creaLibro(actual_file,nom_test," las egunda derivacion es caracterizacion con todo: moho intra y entercd sin reducir na")
    print str("ddd")

    #creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")
except:
    pass


'''

'''
## mohaeavioiaoaus reduced
try:
    actual_file = ['resultatsEntrenament/predict_no_no_norm_mohoeavious_no_reduced_covarNorm_C1','resultatsEntrenament/predict_no_no_norm_mohoeavious_no_reduced_covarNorm_C2','resultatsEntrenament/predict_no_no_norm_mohoeavious_no_reduced_covarNorm_C3']
    nom_test = 'predict_no_no_norm_mohoeavious_NO_reduced_NormalsCovar.xls'
    creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas + mohoeavious (covarianza normales) y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")
except:
    pass
'''

## mohaeavioiaoaus excluyendo los casos que tocan las pelotas

#actual_file = ['resultatsEntrenament/mohoeavious_restricted_casos_rarosC1','resultatsEntrenament/mohoeavious_restricted_casos_rarosC2','resultatsEntrenament/mohoeavious_restricted_casos_rarosC3']
#nom_test = 'predict_no_no_norm_mohoeavious_reduced_NormalsCovar_foraCasosRaros.xls'
#creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas + mohoeavious y SVM, caracteristicas reducidas y covarianza normales. Se entrena sin casos raros '107-00584','107-00679','107-00698','107-00652','107-00697','107-00636'")
#creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")


'''
## mohaeavioiaoaus no reduced EL MEJOR!!!
try:
    actual_file = ['resultatsEntrenament/predict_no_no_norm_mohoeavious_allCovar_C1','resultatsEntrenament/predict_no_no_norm_mohoeavious_allCovar_C2','resultatsEntrenament/predict_no_no_norm_mohoeavious_allCovar_C3']
    nom_test = 'extraccion de predict_no_no_norm_mohoeavious_no_reduced_allCovar normalizadas.xls'
    #creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas + mohoeavious no normalizadas y SVM")
    zz = creaOrLogic1(actual_file[:2],nom_test," mediante extraccion de caracteristicas normalizadas y SVM XX")
except:
    pass
'''

'''
#predict_no_caracteritzadorscov.normal-26-06-2012_covAll_C1
actual_file = ['resultatsEntrenament/predict_no_componentesPruebaJoseDef_covNorm_no_norm_signal-26-06-2012_C1','resultatsEntrenament/predict_no_componentesPruebaJoseDef_covNorm_no_norm_signal-26-06-2012_C2','resultatsEntrenament/predict_no_componentesPruebaJoseDef_covNorm_no_norm_signal-26-06-2012_C3']
nom_test = 'extraccion de _lead1_4_componentesPruebaJoseDef_covNorm_no_norm_signal-28-6-2012.xls'
creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas + mohoeavious no normalizadas y SVM-28-6-2012 (covarainzas normales, senal no norm)")
zz = creaOrLogic1(actual_file[:2],nom_test," mediante extraccion de caracteristicas normalizadas y SVM -28-6-2012(covarainzas normales, senal no norm)")
'''


'''
#intento desesperado 28/06/2012
actual_file = ['predict_no_intent_desesperat-26-06-2012_C1','predict_no_intent_desesperat-26-06-2012_C2','predict_no_intent_desesperat-26-06-2012_C3']
nom_test = 'extraccion de _lead1_4_intent_desesperat-26-06-2012.xls'
creaLibro(actual_file,nom_test," intento desquiciado")
zz = creaOrLogic1(actual_file[:2],nom_test," mediante intento desesperado")
'''

'''

## mohaeavioiaoaus reduced
try:
    actual_file = ['resultatsEntrenament/predict_no_no_norm_mohoeavious_reduced_covarNorm_C1','resultatsEntrenament/predict_no_no_norm_mohoeavious_reduced_covarNorm_C2','resultatsEntrenament/predict_no_no_norm_mohoeavious_reduced_covarNorm_C3']
    nom_test = 'predict_no_no_norm_mohoeavious_reduced_NormalsCovar.xls'
    creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas + mohoeavious y SVM, caracteristicas reducidas y covarianza normales")
    #creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")
except:
    pass




## mohaeavioiaoaus reduced
try:
    actual_file = ['resultatsEntrenament/predict_no_no_norm_mohoeavious_reduced_allCovar_C1','resultatsEntrenament/predict_no_no_norm_mohoeavious_reduced_allCovar_C2','resultatsEntrenament/predict_no_no_norm_mohoeavious_reduced_allCovar_C3']
    nom_test = 'predict_no_no_norm_mohoeavious_reduced_allCovar.xls'
    creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas + mohoeavious y SVM, caracteristicas reducidas ")
    #creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")
except:
    pass

## mohaeavioiaoaus no reduced EL MEJOR!!!
try:
    actual_file = ['resultatsEntrenament/predict_no_no_norm_mohoeavious_allCovar_C1','resultatsEntrenament/predict_no_no_norm_mohoeavious_allCovar_C2','resultatsEntrenament/predict_no_no_norm_mohoeavious_allCovar_C3']
    nom_test = 'extraccion de predict_no_no_norm_mohoeavious_no_reduced_allCovar normalizadas.xls'
    creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas + mohoeavious no normalizadas y SVM")
    #creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")
except:
    pass


## normalizado
try:
    actual_file = ['resultatsEntrenament/predict_norm_all_C1','resultatsEntrenament/predict_norm_all_C2','resultatsEntrenament/predict_norm_all_C3']
    nom_test = 'extraccion de caracteristicas normalizadas.xls'
    creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas normalizadas y SVM ")
except:
    pass

try:
    ##nonormalizado
    actual_file = ['resultatsEntrenament/predict_no_norm_all_C1','resultatsEntrenament/predict_no_norm_all_C2','resultatsEntrenament/predict_no_norm_all_C3']
    nom_test = 'extraccion de caracteristicas no normalizadas.xls'
    creaLibro(actual_file,nom_test," mediante extraccion de caracteristicas no normalizadas y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante extraccion de caracteristicas no normalizadas y SVM ")
except:
    pass


try:    
    ##waveforms no norm
    actual_file = ['resultatsEntrenament/predict_no_norm_resumed_balanced_lineal_C1','resultatsEntrenament/predict_no_norm_resumed_balanced_lineal_C2','resultatsEntrenament/predict_no_norm_resumed_balanced_lineal_C3']
    nom_test = 'entrenamiento reducido.xls'
    creaLibro(actual_file,nom_test,"mediante extraccion de caracteristicas no normalizadas y SVM, entrenamiento reducido")
    #creaOrLogic1(actual_file,nom_test," mediante morfologia no normalizado, PCA14, FS6 y SVM ")
except:
    pass

try:    
    ##waveforms no norm
    actual_file = ['resultatsEntrenament/predict_waveform_no_norm_PCA14_FS6_C1','resultatsEntrenament/predict_waveform_no_norm_PCA14_FS6_C2','resultatsEntrenament/predict_waveform_no_norm_PCA14_FS6_C3']
    nom_test = 'extraccion de caracteristicas waveform no normalizadas.xls'
    creaLibro(actual_file,nom_test," mediante morfologia no normalizado, PCA14, FS6 y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante morfologia no normalizado, PCA14, FS6 y SVM ")
except:
    pass


try:
    ##waveforms norm
    actual_file = ['resultatsEntrenament/predict_waveform_norm_PCA14_FS6_C1','resultatsEntrenament/predict_waveform_norm_PCA14_FS6_C2','resultatsEntrenament/predict_waveform_norm_PCA14_FS6_C3']
    nom_test = 'extraccion de caracteristicas waveform normalizadas.xls'
    creaLibro(actual_file,nom_test," mediante morfologia normalizado, PCA14, FS6 y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante morfologia normalizado, PCA14, FS6 y SVM ")
except:
    pass

try:
    ##waveforms norm simple PCA
    actual_file = ['resultatsEntrenament/predict_waveform_norm_PCA6_C1','resultatsEntrenament/predict_waveform_norm_PCA6_C2','resultatsEntrenament/predict_waveform_norm_PCA6_C3']
    nom_test = 'extraccion de caracteristicas waveform normalizada solo PCA.xls'
    creaLibro(actual_file,nom_test," mediante morfologia normalizado, PCA6 y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante morfologia normalizado, PCA6 y SVM ")
except:
    pass
'''

try:    
    ##waveforms no norm
    actual_file = ['resultatsEntrenament/predict_waveform_no_norm_PCA14_FS6_C1','resultatsEntrenament/predict_waveform_no_norm_PCA14_FS6_C2','resultatsEntrenament/predict_waveform_no_norm_PCA14_FS6_C3']
    nom_test = 'extraccion de caracteristicas waveform no normalizadas.xls'
    creaLibro(actual_file,nom_test," mediante morfologia no normalizado, PCA14, FS6 y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante morfologia no normalizado, PCA14, FS6 y SVM ")
except:
    pass


try:
    ##waveforms norm
    actual_file = ['resultatsEntrenament/predict_waveform_norm_PCA14_FS6_C1','resultatsEntrenament/predict_waveform_norm_PCA14_FS6_C2','resultatsEntrenament/predict_waveform_norm_PCA14_FS6_C3']
    nom_test = 'extraccion de caracteristicas waveform normalizadas.xls'
    creaLibro(actual_file,nom_test," mediante morfologia normalizado, PCA14, FS6 y SVM ")
    #creaOrLogic1(actual_file,nom_test," mediante morfologia normalizado, PCA14, FS6 y SVM ")
except:
    pass

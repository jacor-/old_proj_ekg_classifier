
from tempfile import TemporaryFile
from xlwt import Workbook, easyxf, BIFFRecords


'''
row1 = sheet2.row(1)
row1.write(0,'A2')
row1.write(1,'B2')
sheet2.col(0).width = 10000
sheet2 = book.get_sheet(1)
sheet2.row(0).write(0,'Sheet 2 A1')
sheet2.row(0).write(1,'Sheet 2 B1')
sheet2.flush_row_data()
sheet2.write(1,0,'Sheet 2 A3')
sheet2.col(0).width = 5000
sheet2.col(0).hidden = True
'''

def getLeads():
    return ['C1','C2','C3']

def getMethods():
    #return ['Cardios Manager', 'Pasos por cero', 'Sin umbrales', 'Umbral adaptativo', 'Umbral adaptativo Hilbert', 'Umbral adaptativo Haar']
    return ['qrsAdHaar', 'zeroStep', 'sample', 'qrsAd', 'qrsAdHilb', 'ref']


def getCases():
    return ['107-00703',
             '107-00702',
             '107-00701',
             '107-00700',
             '107-00707',
             '107-00706',
             '107-00705',
             '107-00709',
             '107-00708',
             '107-00654',
             '107-00657',
             '107-00656',
             '107-00652',
             '107-00659',
             '107-00658',
             '107-00710',
             '107-00558',
             '107-00559',
             '107-00555',
             '107-00642',
             '107-00640',
             '107-00649',
             '107-00744',
             '107-00563',
             '107-00679',
             '107-00560',
             '107-00567',
             '107-00566',
             '107-00564',
             '107-00673',
             '107-00672',
             '107-00671',
             '107-00568',
             '107-00677',
             '107-00676',
             '107-00675',
             '107-00674',
             '107-00748',
             '107-00575',
             '107-00576',
             '107-00668',
             '107-00571',
             '107-00573',
             '107-00665',
             '107-00666',
             '107-00667',
             '107-00663',
             '107-00691',
             '107-00690',
             '107-00693',
             '107-00692',
             '107-00694',
             '107-00697',
             '107-00699',
             '107-00698',
             '107-00747',
             '107-00746',
             '107-00678',
             '107-00743',
             '107-00741',
             '107-00610',
             '107-00613',
             '107-00614',
             '107-00616',
             '107-00581',
             '107-00583',
             '107-00585',
             '107-00584',
             '107-00682',
             '107-00683',
             '107-00680',
             '107-00681',
             '107-00686',
             '107-00687',
             '107-00684',
             '107-00685',
             '107-00688',
             '107-00689',
             '107-00602',
             '107-00603',
             '107-00600',
             '107-00606',
             '107-00607',
             '107-00605',
             '107-00599',
             '107-00592',
             '107-00593',
             '107-00596',
             '107-00597',
             '107-00729',
             '107-00728',
             '107-00725',
             '107-00726',
             '107-00722',
             '107-00639',
             '107-00636',
             '107-00634',
             '107-00631',
             '107-00630',
             '107-00738',
             '107-00739',
             '107-00736',
             '107-00737',
             '107-00734',
             '107-00735',
             '107-00732',
             '107-00731',
             '107-00620',
             '107-00621',
             '107-00622',
             '107-00624',
             '107-00626',
             '107-00628']


def getResults(case, lead, method, aa):

    c = aa[case][method][lead]
    from numpy import nan
            
    return {'TP':c[0],
                'FP':c[1],
                'FN':c[2],
                'Sensibility': c[3],
                'VPP': c[4]
                }

f = open('/home/jose/workspace/holter_analysis/src/resultatsDeteccio2052012.res','rb')
import cPickle
whole_results = cPickle.load(f)           

book = Workbook()

sheet = book.add_sheet('full N')
sheet.portrait = True
sheet.left_margin = 0.25
sheet.right_margin = 0.25
sheet.top_margin = 0.25
sheet.bottom_margin = 0.25


st_value_letra_gorda = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 130;');
st_value = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
st_value_right = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
st_value_down = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
st_value_corner = easyxf('align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');
st_value_senspec = easyxf('pattern: pattern solid, fore_colour pale_blue;''align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 110;');

style_title = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''borders: left hair, right hair, top hair, bottom hair;''font: height 150;')
style2_title_vert = easyxf('pattern: pattern solid, fore_colour gray_ega;''align: vertical center, horizontal center;''alignment: rotation 90;''borders: left hair, right hair, top hair, bottom hair;''font: height 150;')
style_none = easyxf('align: vertical center, horizontal center;''borders: left  no_line, right  no_line, top  no_line, bottom  no_line;''font: height 150;')


sheet.write(1,0,'Caso',style = style_title)
sheet.write(1,1,'Beats',style = style_title)
sheet.write(1,2,'Lead',style = style_title)

sheet.col(0).width = 2000
sheet.col(1).width = 1600
sheet.col(2).width = 1600

st = 3
for metode in getMethods():
    sheet.col(st).width = 1300
    sheet.col(st+1).width = 1000
    sheet.col(st+2).width = 1000

    sheet.write_merge(0,0,st,st+2,metode,style = style_title)

    sheet.write(1,st,"TP",style = style_title)
    sheet.write(1,st+1,"FP",style = style_title)
    sheet.write(1,st+2,"FN",style = style_title)
    
    st = st + 3    


from numpy import zeros
resumen_total = zeros([3,40])


st_raw = 2
st_col = 3

for case in getCases():
    sheet.write_merge(st_raw,st_raw+2,0,0,case,style = style_title)        
    for index_lead, lead in enumerate(getLeads()):
        st_col = 3
        sheet.write(st_raw+index_lead,1,lead,style = style_title)
        for index_method, method in enumerate(getMethods()):
            if index_method == 0 and index_lead == 0:
                a = getResults(case, index_lead, method, whole_results)
                a = int(a['TP']+a['FN'])
                sheet.write_merge(st_raw,st_raw+2,2,2,str(a),style = st_value)
                resumen_total[0][2]=resumen_total[0][2]+a
                resumen_total[1][2]=resumen_total[1][2]+a
                resumen_total[2][2]=resumen_total[2][2]+a
            a = getResults(case, index_lead, method, whole_results)
            sheet.write(st_raw+index_lead,st_col,str(int(a['TP'])),style = st_value)
            resumen_total[index_lead][st_col]=resumen_total[index_lead][st_col]+a['TP']
            sheet.write(st_raw+index_lead,st_col+1,str(int(a['FP'])),style = st_value)
            resumen_total[index_lead][st_col+1]=resumen_total[index_lead][st_col+1]+a['FP']
            sheet.write(st_raw+index_lead,st_col+2,str(int(a['FN'])),style = st_value)
            resumen_total[index_lead][st_col+2]=resumen_total[index_lead][st_col+2]+a['FN']
            st_col = st_col + 3
    st_raw = st_raw + 3
    


sheet.write_merge(st_raw,st_raw+2,0,0,"Resumen\npor\nderivacion",style = style_title)
sheet.write_merge(st_raw,st_raw+2,2,2,str(int(resumen_total[0][2])),style = st_value)
for index_derivacion_resumen,derivacion_resum in enumerate(resumen_total):
    sheet.write(st_raw+index_derivacion_resumen,1,getLeads()[index_derivacion_resumen],style = style_title)
    

    for index_tot, elem_tot in enumerate(derivacion_resum):
        if index_tot != 2:
            if elem_tot != 0:
                sheet.write(st_raw+index_derivacion_resumen,index_tot,elem_tot,style = st_value)
st_raw = st_raw + 3


sheet.write_merge(st_raw,st_raw,0,1,"Resumen general",style = style_title)

from numpy import sum
resum_extrem = sum(resumen_total,0)
for index_tot, elem_tot in enumerate(resum_extrem):
    if elem_tot != 0:
        sheet.write(st_raw,index_tot,elem_tot,style = st_value)










st_raw = 0
sheet2 = book.add_sheet('sensibilityUndVpp')

sheet2.write_merge(st_raw,st_raw,0,0,"Sensibility",style = style_title)

sheet2.col(0).width = 2000

sheet2.write(1,0,'C1',style = style_title)
sheet2.write(2,0,'C2',style = style_title)
sheet2.write(3,0,'C3',style = style_title)
sheet2.write(4,0,'mean',style = style_title)
sheet2.write(5,0,'std',style = style_title)

from numpy import around,mean,std

st = 1
for index_method, metode in enumerate(getMethods()):
    sheet2.col(st).width = 2000
    sheet2.write(0,st,metode,style = style_title)
    vals = []
    for lead_index in range(3):
        sens = resumen_total[lead_index][3+index_method*3:3+index_method*3+3]
        sens = float(sens[0])/(sens[0]+sens[2])

        sheet2.write(lead_index+1,st,str(float(around(sens*100,2))),style =st_value_letra_gorda)
        vals.append(sens)
    
    sheet2.write(2+2,st,str(float(around(mean(vals)*100,2))),style =st_value_letra_gorda)
    sheet2.write(2+3,st,str(float(around(std(vals),4))),style =st_value_letra_gorda)
        
    st = st + 1    
    


offset = 8

sheet2.write_merge(st_raw+offset,st_raw+offset,0,0,"VPP",style = style_title)

sheet2.write(1+offset,0,'C1',style = style_title)
sheet2.write(2+offset,0,'C2',style = style_title)
sheet2.write(3+offset,0,'C3',style = style_title)
sheet2.write(4+offset,0,'mean',style = style_title)
sheet2.write(5+offset,0,'std',style = style_title)


from numpy import around

st = 1
for index_method, metode in enumerate(getMethods()):
    sheet2.write(0+offset,st,metode,style = style_title)
    vals = []
    for lead_index in range(3):
        sens = resumen_total[lead_index][3+index_method*3:3+index_method*3+3]
        sens = float(sens[0])/(sens[0]+sens[1])

        sheet2.write(lead_index+1+offset,st,str(float(around(sens*100,2))), style =st_value_letra_gorda)
        vals.append(sens)
    
    sheet2.write(lead_index+2+offset,st,str(float(around(mean(vals)*100,2))),style =st_value_letra_gorda)
    sheet2.write(lead_index+3+offset,st,str(float(around(std(vals),4))),style =st_value_letra_gorda)
    st = st + 1










f.close()



'''
sheet.write_merge(1,2,1,1,'id_cas',style = style_title)
sheet.write_merge(1,2,2,2,'lead',style = style_title)
sheet.write_merge(1,2,3,3,'method',style = style_title)
sheet.col(1).width = 800
sheet.col(2).width = 800

sheet.write_merge(1,1,4,9,"normales", style = style2_title_vert)
sheet.write_merge(1,1,10,15,"ventriculares",style = style2_title_vert)
sheet.write(2,4,'TP',style = style_title)
sheet.write(2,5,'FP',style = style_title)
sheet.write(2,6,'TN',style = style_title)
sheet.write(2,7,'FN',style = style_title)
sheet.write(2,8,'Sens',style = style_title)
sheet.write(2,9,'Spec',style = style_title)
sheet.write(2,10,'TP',style = style_title)
sheet.write(2,11,'FP',style = style_title)
sheet.write(2,12,'TN',style = style_title)
sheet.write(2,13,'FN',style = style_title)
sheet.write(2,14,'Sens',style = style_title)
sheet.write(2,15,'Spec',style = style_title)

for i in range(19):
    sheet.row(i).height = 200


left_align = 3
sheet.row(3).height = 400

for ind_case, case in enumerate(getCases()[:]):
    print(case + "    " + str(ind_case))
    AMPLE_CASE = len(getLeads())*len(getMethods())
    sheet.write_merge(left_align,left_align+AMPLE_CASE-1,1,1,case,style = style_title)
    for lead_ind,lead in enumerate(getLeads()):
        AMPLE_LEAD = len(getMethods())
        
        sheet.write_merge(left_align,left_align+AMPLE_LEAD-1,2,2,lead,style = style_title)
        for met_ind,method in enumerate(getMethods()):
            if met_ind == len(getMethods())-1 and lead_ind == len(getLeads())-1:
                last_style = st_value_corner
                nrm_style = st_value_right
            else:
                last_style = st_value_down
                nrm_style = st_value

            sheet.col(left_align).width = 800

            sheet.write(left_align,3,method,style = style2_title_vert)
            vertical = 4
            res = getResults(case, lead, method)
            
                        
            V = res['N']
            sheet.write(left_align,vertical,str(V['TP']),style = nrm_style)
            sheet.write(left_align,vertical+1,str(V['FP']),style = nrm_style)
            sheet.write(left_align,vertical+2,str(V['TN']),style = nrm_style)
            sheet.write(left_align,vertical+3,str(V['FN']),style = nrm_style)
            sheet.write(left_align,vertical+4,str(V['Sensibility']),style = st_value_senspec)
            sheet.write(left_align,vertical+5,str(V['Specificity']),style = st_value_senspec)
            V = res['V']
            sheet.write(left_align,vertical+6,str(V['TP']),style = nrm_style)
            sheet.write(left_align,vertical+7,str(V['FP']),style = nrm_style)
            sheet.write(left_align,vertical+8,str(V['TN']),style = nrm_style)
            sheet.write(left_align,vertical+9,str(V['FN']),style = nrm_style)
            sheet.write(left_align,vertical+10,str(V['Sensibility']),style = st_value_senspec)
            sheet.write(left_align,vertical+11,str(V['Specificity']),style = st_value_senspec)
            
            
            
            left_align = left_align + 1


for i in range(left_align+1):
    sheet.write(i,0," ",style = style_none)

''' 
            
book.save('simple.xls')
book.save(TemporaryFile())

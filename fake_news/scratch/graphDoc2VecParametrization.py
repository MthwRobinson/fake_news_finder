import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#WINDOWS * FEATURES
#windows = [1,2,3, 4, 6, 10, 20, 50, 100]
#sizes = [10, 25, 50, 100, 300, 500, 750, 1000, 2000]
#raw_results = [[[0.5943396226415094, 0.5927099841521395, 0.5887850467289719, 0.6276923076923077, 0.5851393188854489], [0.6060606060606061, 0.6106623586429725, 0.6106623586429726, 0.6293494704992435, 0.6290076335877863], [0.5873261205564142, 0.6015037593984962, 0.5973645680819911, 0.5940902021772939, 0.571875], [0.5596026490066225, 0.57984496124031, 0.6084243369734789, 0.5624012638230649, 0.5972006220839814], [0.5607779578606158, 0.6041335453100158, 0.5575959933222037, 0.5760000000000001, 0.6125], [0.5623003194888179, 0.608, 0.6052227342549923, 0.5902668759811617, 0.6263565891472869], [0.57001647446458, 0.5727999999999999, 0.5632911392405064, 0.5820433436532507, 0.62015503875969], [0.5700636942675158, 0.5938461538461538, 0.543778801843318, 0.5887850467289719, 0.5955555555555556], [0.5871271585557299, 0.5525902668759811, 0.5472312703583063, 0.6049382716049383, 0.6034755134281201]], [[0.5802269043760129, 0.5960912052117264, 0.5774193548387097, 0.5958795562599051, 0.6195826645264848], [0.6018518518518519, 0.5907692307692308, 0.5680672268907563, 0.5952755905511812, 0.5945072697899838], [0.606741573033708, 0.621875, 0.613251155624037, 0.6287519747235386, 0.6424242424242423], [0.631083202511774, 0.6437499999999999, 0.5774877650897227, 0.61875, 0.5917721518987342], [0.6125, 0.6121112929623569, 0.5945121951219512, 0.592948717948718, 0.6358024691358025], [0.609105180533752, 0.6363636363636365, 0.6018808777429467, 0.6193353474320241, 0.6170542635658915], [0.6432926829268293, 0.6182380216383307, 0.633693972179289, 0.6319018404907976, 0.6151468315301392], [0.6134969325153375, 0.606060606060606, 0.5977382875605816, 0.6022187004754358, 0.6141732283464567], [0.6265822784810127, 0.6319115323854659, 0.5964912280701754, 0.61875, 0.6456692913385826]], [[0.6, 0.6048, 0.5812807881773399, 0.6015748031496062, 0.6200607902735562], [0.5944170771756979, 0.5831960461285007, 0.5562913907284769, 0.6099071207430341, 0.6102236421725239], [0.6372549019607843, 0.6214511041009464, 0.6378205128205129, 0.6042003231017771, 0.6655896607431341], [0.5666666666666665, 0.6232114467408586, 0.5860927152317881, 0.6382306477093208, 0.6288492706645057], [0.6272727272727273, 0.6260032102728732, 0.5971563981042655, 0.60625, 0.6163522012578617], [0.6330275229357798, 0.6537890044576522, 0.6224961479198768, 0.6813509544787079, 0.6775147928994083], [0.5714285714285713, 0.6025236593059937, 0.5766666666666665, 0.576923076923077, 0.65], [0.5865384615384615, 0.5933014354066987, 0.5815831987075929, 0.5889967637540454, 0.5981308411214953], [0.6126984126984126, 0.605475040257649, 0.5867098865478121, 0.6161290322580646, 0.6466666666666666]], [[0.5980707395498391, 0.6038338658146964, 0.6062500000000001, 0.6160990712074305, 0.6366459627329193], [0.6298701298701299, 0.5903225806451612, 0.6336633663366337, 0.6540880503144654, 0.6307448494453248], [0.6048780487804878, 0.617246596066566, 0.5895765472312703, 0.58678955453149, 0.625], [0.5984496124031008, 0.6165884194053208, 0.5899513776337115, 0.589171974522293, 0.6379044684129429], [0.5802269043760129, 0.6384976525821596, 0.5579119086460033, 0.5866666666666667, 0.6038961038961038], [0.592948717948718, 0.6132075471698114, 0.570957095709571, 0.605475040257649, 0.6281249999999999], [0.5933014354066986, 0.6163723916532905, 0.5880503144654088, 0.5546492659053833, 0.6380368098159509], [0.5691823899371069, 0.5723684210526317, 0.5617977528089887, 0.5570032573289903, 0.6052227342549924], [0.6088379705400981, 0.6255778120184899, 0.6071428571428572, 0.573743922204214, 0.5664000000000001]], [[0.5895765472312704, 0.609375, 0.5760000000000001, 0.5391014975041597, 0.6197183098591549], [0.6000000000000001, 0.5861513687600644, 0.5878594249201278, 0.6139240506329113, 0.6130030959752322], [0.6174055829228243, 0.6237942122186495, 0.6196721311475409, 0.6351791530944625, 0.6525974025974026], [0.6106623586429726, 0.6415662650602408, 0.6114649681528663, 0.6136724960254372, 0.6200317965023848], [0.6220839813374806, 0.5801282051282051, 0.591869918699187, 0.60828025477707, 0.5935483870967742], [0.6237942122186495, 0.6258064516129032, 0.6121112929623568, 0.6003316749585406, 0.6418152350081036], [0.6252100840336134, 0.6440129449838188, 0.6032786885245902, 0.6099999999999999, 0.6327503974562798], [0.5990338164251208, 0.5833333333333334, 0.6006289308176102, 0.5945072697899838, 0.6370597243491577], [0.6304347826086956, 0.6025236593059937, 0.5785381026438569, 0.6066066066066066, 0.6216640502354788]], [[0.610223642172524, 0.5980392156862745, 0.619047619047619, 0.6097946287519747, 0.625], [0.6065573770491804, 0.6307692307692309, 0.6048000000000001, 0.6295707472178059, 0.6330708661417322], [0.6296900489396412, 0.6305732484076434, 0.5690515806988352, 0.6166394779771616, 0.6529968454258674], [0.5987460815047021, 0.6280487804878049, 0.6656394453004624, 0.6160990712074305, 0.6189735614307932], [0.6296296296296297, 0.6268174474959611, 0.6291079812206573, 0.6126984126984127, 0.639871382636656], [0.6064516129032259, 0.6276276276276277, 0.6161137440758294, 0.641390205371248, 0.6475279106858054], [0.6035313001605137, 0.6016, 0.59, 0.6079734219269102, 0.617363344051447], [0.6163934426229508, 0.5785953177257525, 0.6015999999999999, 0.5845648604269293, 0.6230031948881789], [0.6, 0.6234177215189873, 0.6131621187800963, 0.6178343949044586, 0.6281249999999999]], [[0.5856905158069883, 0.6257861635220127, 0.604200323101777, 0.56682769726248, 0.6129032258064515], [0.6061588330632091, 0.6153846153846153, 0.6108527131782946, 0.6236220472440944, 0.6470588235294118], [0.5746031746031747, 0.5723370429252782, 0.5893719806763285, 0.6038338658146966, 0.5923664122137405], [0.6209048361934478, 0.61875, 0.5971107544141253, 0.6492307692307693, 0.6314152410575428], [0.6023294509151413, 0.6171617161716172, 0.6052631578947368, 0.6339869281045751, 0.6163934426229507], [0.5996810207336524, 0.6507936507936508, 0.6425196850393701, 0.6467817896389324, 0.6391096979332274], [0.6175548589341693, 0.6124401913875598, 0.6196721311475409, 0.6148969889064976, 0.6494345718901454], [0.6270627062706271, 0.6115702479338843, 0.6218487394957983, 0.6305418719211824, 0.6401273885350319], [0.6153846153846153, 0.6464968152866243, 0.618066561014263, 0.6085526315789473, 0.6634460547504026]], [[0.6377295492487479, 0.6455696202531646, 0.6209150326797386, 0.6560509554140128, 0.6539074960127592], [0.6192733017377569, 0.6101694915254238, 0.6047244094488189, 0.6397515527950312, 0.6380368098159509], [0.6307448494453249, 0.615146831530139, 0.59672131147541, 0.6277602523659306, 0.6235864297253635], [0.6213592233009709, 0.6132075471698114, 0.6114649681528663, 0.6064516129032257, 0.6509433962264151], [0.6071428571428571, 0.6343042071197412, 0.6507936507936507, 0.6191247974068071, 0.6332794830371566], [0.6177847113884555, 0.624203821656051, 0.6233333333333334, 0.6420545746388444, 0.6467817896389325], [0.5983739837398374, 0.5746388443017656, 0.5821138211382113, 0.5786163522012578, 0.6475279106858054], [0.5884194053208137, 0.6026490066225164, 0.6216640502354789, 0.6277602523659306, 0.6236220472440944], [0.5927099841521395, 0.6335403726708074, 0.5852090032154341, 0.6437499999999999, 0.6415094339622641]], [[0.5983739837398374, 0.6289308176100628, 0.6013513513513514, 0.6366666666666667, 0.6437908496732028], [0.6250000000000001, 0.6052631578947368, 0.5690515806988352, 0.6151419558359621, 0.5898305084745763], [0.5894378194207837, 0.6288492706645056, 0.6288659793814432, 0.6010016694490817, 0.6217105263157895], [0.6146179401993356, 0.6299999999999999, 0.584717607973422, 0.5714285714285715, 0.6307448494453248], [0.5644371941272431, 0.5973597359735974, 0.6112956810631229, 0.610223642172524, 0.5942492012779552], [0.6031746031746031, 0.5695364238410596, 0.5891980360065465, 0.5893719806763285, 0.6464], [0.5806451612903226, 0.646875, 0.6146645865834635, 0.624405705229794, 0.5977382875605816], [0.6310679611650485, 0.5939597315436241, 0.6184210526315789, 0.6085578446909669, 0.6442307692307692], [0.6105610561056105, 0.6273885350318471, 0.6223662884927067, 0.6379585326953748, 0.6003316749585407]]]

#MINCOUNTS * FEATURES
#window = 10
min_counts = [1, 10, 50, 100, 300, 500, 900, 1500]
sizes = [50, 500, 750, 1000, 2000, 3000, 5000]
raw_results = [[[0.5996810207336524, 0.60882800608828, 0.5950155763239876, 0.6067415730337079, 0.6220839813374806], [0.5793780687397708, 0.5733558178752108, 0.5784313725490197, 0.6042003231017771, 0.6226993865030676], [0.6085578446909667, 0.6015503875968993, 0.6203554119547657, 0.5899513776337116, 0.6199021207177815], [0.6050420168067228, 0.6329113924050632, 0.5560975609756097, 0.6035889070146819, 0.6136363636363636], [0.6085526315789473, 0.6405023547880692, 0.6069182389937106, 0.6053882725832014, 0.6393700787401575], [0.6099518459069022, 0.5714285714285714, 0.559463986599665, 0.5896147403685091, 0.6330275229357799], [0.6379585326953748, 0.6314102564102564, 0.6199021207177814, 0.6003210272873194, 0.6222910216718266]], [[0.5974025974025974, 0.6113671274961597, 0.570480928689884, 0.603225806451613, 0.625194401244168], [0.618657937806874, 0.6214511041009464, 0.5833333333333333, 0.6166394779771616, 0.6740858505564388], [0.5741935483870967, 0.6260296540362439, 0.5833333333333333, 0.5925925925925926, 0.6009693053311792], [0.6033333333333334, 0.6256239600665557, 0.5979020979020979, 0.564625850340136, 0.6022187004754358], [0.6243902439024389, 0.6246056782334385, 0.5964343598055105, 0.6290322580645161, 0.6322580645161291], [0.5937499999999999, 0.6350710900473934, 0.5876623376623376, 0.5903225806451613, 0.6275752773375595], [0.6047297297297297, 0.5893719806763285, 0.5950413223140496, 0.6075949367088608, 0.6307448494453248]], [[0.6257861635220126, 0.6038961038961038, 0.59672131147541, 0.6043613707165109, 0.63125], [0.5814696485623003, 0.5927099841521395, 0.5645161290322581, 0.60828025477707, 0.64251968503937], [0.653250773993808, 0.6403940886699507, 0.6234177215189872, 0.6312399355877616, 0.6292134831460675], [0.631578947368421, 0.6288492706645056, 0.6092503987240829, 0.6116504854368933, 0.6360759493670886], [0.5639344262295082, 0.5888000000000001, 0.5742574257425742, 0.5844155844155844, 0.6112852664576803], [0.6325878594249201, 0.6719242902208202, 0.6432, 0.5922330097087378, 0.6559485530546624], [0.6075533661740559, 0.6141732283464566, 0.5551839464882943, 0.5963756177924219, 0.6222222222222222]], [[0.6580645161290323, 0.6268174474959611, 0.6072607260726073, 0.6310679611650486, 0.6727828746177369], [0.6526655896607431, 0.6032258064516128, 0.5886178861788618, 0.6666666666666667, 0.624223602484472], [0.5912162162162162, 0.6421725239616614, 0.6003262642740621, 0.608, 0.6362204724409449], [0.6062602965403625, 0.637223974763407, 0.5853658536585366, 0.61875, 0.624223602484472], [0.6308943089430894, 0.6191247974068071, 0.6538461538461539, 0.6239737274220033, 0.6019417475728155], [0.5868465430016864, 0.608130081300813, 0.6153846153846153, 0.5943238731218697, 0.6387096774193548], [0.5897435897435898, 0.6125, 0.5851239669421487, 0.569182389937107, 0.6064516129032258]], [[0.5839874411302983, 0.6251944012441679, 0.6021840873634946, 0.6061538461538462, 0.627450980392157], [0.608130081300813, 0.6111111111111112, 0.6265822784810127, 0.5895061728395061, 0.6718266253869969], [0.5812807881773399, 0.6018808777429467, 0.6136724960254372, 0.5793780687397709, 0.631578947368421], [0.56, 0.6188197767145135, 0.5771144278606964, 0.6096774193548387, 0.6068515497553018], [0.6030150753768845, 0.6431999999999999, 0.6270627062706271, 0.635483870967742, 0.655683690280066], [0.5884297520661157, 0.6095551894563426, 0.6053511705685618, 0.5867098865478121, 0.6191247974068071], [0.5797101449275361, 0.6389776357827476, 0.5919732441471572, 0.6107594936708861, 0.6003316749585407]], [[0.6019108280254777, 0.61198738170347, 0.5993485342019543, 0.6309148264984228, 0.6251993620414673], [0.5657237936772047, 0.5938009787928221, 0.5647840531561462, 0.6217457886676876, 0.6656626506024096], [0.6006493506493507, 0.6498422712933754, 0.6096423017107309, 0.6275115919629057, 0.6543778801843317], [0.5949367088607594, 0.6359300476947536, 0.6115702479338844, 0.6613924050632912, 0.6488188976377952], [0.5826513911620295, 0.60828025477707, 0.573743922204214, 0.619047619047619, 0.6527999999999999], [0.6091205211726384, 0.6454689984101749, 0.6054421768707483, 0.6200317965023847, 0.641860465116279], [0.6128500823723229, 0.6240000000000001, 0.5851239669421487, 0.6123778501628664, 0.6186579378068738]], [[0.6232114467408586, 0.6022544283413849, 0.578512396694215, 0.5728000000000001, 0.6282051282051281], [0.6129032258064516, 0.5987261146496814, 0.6320907617504051, 0.6280193236714975, 0.6310679611650486], [0.5631067961165049, 0.5805422647527911, 0.5848142164781905, 0.6022544283413849, 0.6061588330632091], [0.6335403726708075, 0.625, 0.6012461059190032, 0.5869218500797448, 0.6346153846153845], [0.5781990521327014, 0.6391437308868502, 0.5983471074380164, 0.6119873817034699, 0.6373983739837399], [0.6033898305084746, 0.6219312602291326, 0.6295707472178059, 0.631911532385466, 0.6173228346456694], [0.6103286384976525, 0.6269592476489029, 0.6089743589743589, 0.6358024691358025, 0.6605783866057838]], [[0.6427457098283932, 0.5741935483870968, 0.6283048211508554, 0.6055900621118013, 0.6573208722741433], [0.609271523178808, 0.6273291925465838, 0.5891719745222931, 0.6346749226006191, 0.6302250803858521], [0.5865384615384615, 0.6219312602291326, 0.5865384615384616, 0.6146341463414634, 0.6210191082802549], [0.6003210272873195, 0.6215780998389694, 0.6138613861386139, 0.6220095693779905, 0.6477093206951027], [0.5594855305466238, 0.5899053627760253, 0.5939968404423381, 0.5858267716535432, 0.6319115323854662], [0.6041335453100158, 0.5980707395498394, 0.594855305466238, 0.6263910969793322, 0.6376811594202898], [0.568595041322314, 0.5728155339805825, 0.5903814262023217, 0.6123778501628664, 0.6192733017377566]]]
results = np.array(raw_results)
results_mean = np.mean(results, 2)

#WINDOWS * FEATURES
#windows_index = pd.Index(data=windows, name="Words window")
#sizes_index = pd.Index(data=sizes, name="# Features")
#df = pd.DataFrame(data=results_mean, columns=windows_index, index=sizes_index)

#MINCOUNTS * FEATURES
min_counts_index = pd.Index(data=min_counts, name="Min counts")
sizes_index = pd.Index(data=sizes, name="# Features")
df = pd.DataFrame(data=results_mean, columns=sizes_index, index=min_counts_index)


sns.set(style="white")
f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df, cmap=cmap,  vmax=0.66, vmin=0.56, ax=ax, annot=True)
ax.set_title("Doc2Vec f-score")
ax.title.set_fontsize(13)
sns.plt.show()
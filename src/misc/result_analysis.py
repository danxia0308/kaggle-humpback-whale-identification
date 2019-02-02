import os
from pandas import read_csv
import shutil

base_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/'
train_csv_path=base_dir+'train.csv'
test_image_name_file=base_dir+'test_list.txt'
train_classed_path=base_dir+'clean_train_classed/'
test_dir=base_dir+'clean_train/'
dst_dir=base_dir+"/analysis1/"

with open(test_image_name_file) as f:
    test_names=f.readline().split(' ')

image_name_2_class_name=dict([(y,z) for x,y,z in read_csv(train_csv_path).to_records()])
class_name_2_image_names_all={}
for image_name in image_name_2_class_name.keys():
    class_name=image_name_2_class_name.get(image_name)
    if class_name not in class_name_2_image_names_all:
        image_names=[]
    else:
        image_names=class_name_2_image_names_all.get(class_name)
    if image_name not in image_names:
        image_names.append(image_name)
    class_name_2_image_names_all[class_name]=image_names

results=[['w_e08506c', 'w_6822dbc', 0.948725635586491], ['w_75d0e61', 'w_22b8752', 0.8174012679817289], ['w_c815410', 'w_2f1488c', 0.9754372454778392], ['w_640a24a', 'w_d8d0dd9', 0.7937643602461848], ['w_c073f96', 'w_e99ed06', 0.9035981487245698], ['w_b5e6c9c', 'w_985d205', 0.8388362979859838], ['w_9bd2794', 'w_67a9841', 0.879581945629198], ['w_cfae8d8', 'w_f3a12bc', 0.9864000301708933], ['w_f276da3', 'w_dea6580', 0.8526423467416171], ['w_33696ac', 'w_a6a732d', 0.9016244976760833], ['w_8431ae8', 'w_cbda0d3', 0.9777587374944392], ['w_5793f1c', 'w_ae8982d', 0.7295260699578376], ['w_b54f70f', 'w_bc7de9f', 0.7817230904390056], ['w_3815890', 'w_ac2c28e', 0.9908360304415602], ['w_659bdb8', 'w_b495218', 0.918586742363968], ['w_a3181a0', 'w_dfe485f', 0.9269114046687695], ['w_70d0b3c', 'w_4661aac', 0.8515504964574726], ['w_d17b0c5', 'w_b5f7e89', 0.8204810244269751], ['w_6fe5b8e', 'w_55a34c7', 1.0874554253147455], ['w_9a9a2c8', 'w_f3f23d6', 0.9657672245211779], ['w_b3ca4b7', 'w_52b60a6', 1.009610204981538], ['w_d0bfef3', 'w_0e0b65c', 0.8166598494116024], ['w_8cfabed', 'w_c481650', 0.8536242133644389], ['w_d980663', 'w_fc6a5a2', 0.8748466081210564], ['w_a059841', 'w_446a02d', 0.8366567022987395], ['w_d6aa3f3', 'w_4c614f3', 0.8852867774785664], ['w_e99e430', 'w_14e5fe2', 0.7895056858210099], ['w_b99f945', 'w_4cbf396', 0.9200926650651964], ['w_4c0a2c9', 'w_e966f36', 0.6072949046649261], ['w_eb13108', 'w_0d43823', 1.004536806031024], ['w_bfcad53', 'w_0bc078c', 1.0167769363609804], ['w_69c06ad', 'w_2926c77', 0.8733392859544696], ['w_d9f8641', 'w_1133530', 0.8767474835928515], ['w_2c94198', 'w_d7aef56', 0.9505195702992986], ['w_11d8c70', 'w_09bd7e8', 0.8513554673177569], ['w_025911c', 'w_78351ee', 0.9295506517579055], ['w_ae6ac74', 'w_95a5270', 0.859849283741442], ['w_34d7623', 'w_dee1053', 1.0160570771885533], ['w_8a6a8d5', 'w_2d6596f', 0.8857946807167987], ['w_1c6465a', 'w_fc10c15', 0.8358247045729119], ['w_9032a74', 'w_61ed07a', 0.8457647761397664], ['w_090c801', 'w_fc10c15', 0.91329014954063], ['w_dee1053', 'w_be4f189', 0.96906295244205], ['w_4f9c015', 'w_db1f3de', 0.9014639945117334], ['w_9d86594', 'w_14ece76', 0.8629103467600336], ['w_6cfa650', 'w_81d4bd6', 0.7941324774356333], ['w_d24c654', 'w_8a9e295', 0.7546041261435027], ['w_32da935', 'w_bf7c89a', 0.866920664972462], ['w_de657c1', 'w_1be76f5', 0.894292120231995], ['w_10f67bb', 'w_955bfe2', 0.6396878974331001], ['w_2d0cfc1', 'w_aee1ccf', 0.8101211704058308], ['w_857cb95', 'w_71fbd51', 1.0090115997409972], ['w_e966f36', 'w_5d06edd', 0.8556835068205423], ['w_fbf7d73', 'w_b7aa4d8', 0.7323500219579082], ['w_9573686', 'w_2cd7341', 0.7098204701508914], ['w_7e56d66', 'w_1ea8997', 0.9462767236447067], ['w_d223b2e', 'w_f48451c', 0.8931340006922801], ['w_f3a12bc', 'w_2b069ba', 0.9103498884182504], ['w_1b2bf0f', 'w_3f8ff7c', 0.806538204966321], ['w_bbfce38', 'w_a47dc2d', 0.7172703729095724], ['w_fa8c44c', 'w_2f8de4f', 0.8987471157729763], ['w_0027efa', 'w_d7aef56', 1.0069889367291975], ['w_61b9586', 'w_857cb95', 0.7295145302651367], ['w_0b398b2', 'w_1032bb6', 1.0203221144498507]]
results=[['w_c815410', 'w_32eac70', 1.0535151889566334], ['w_b9e5911', 'w_065ee1b', 0.82256857300909], ['w_c073f96', 'w_bd9f354', 0.9091227389764454], ['w_b5e6c9c', 'w_0a0c768', 0.7831305499708543], ['w_33696ac', 'w_93fddcb', 0.8997654297485471], ['w_8431ae8', 'w_50fd107', 0.8130225379751242], ['w_5793f1c', 'w_fe8aee2', 0.7634536450428155], ['w_3815890', 'w_b938e96', 0.9303978713124966], ['w_659bdb8', 'w_b495218', 0.8228211701391073], ['w_9a9a2c8', 'w_32de0c5', 0.9899025978127518], ['w_d980663', 'w_7b05eeb', 0.9244378952258425], ['w_a059841', 'w_b0cadc0', 0.9317779611660841], ['w_4c0a2c9', 'w_e966f36', 0.7347875808497558], ['w_eb13108', 'w_02469a1', 0.8173522004444116], ['w_bfcad53', 'w_857cb95', 1.073642575076021], ['w_2c94198', 'w_d7aef56', 0.9122843043659415], ['w_11d8c70', 'w_09bd7e8', 0.9601827213841116], ['w_025911c', 'w_b311fb6', 0.9961795321753031], ['w_ae6ac74', 'w_c8d5ad5', 0.9850444106757208], ['w_34d7623', 'w_ab44ae4', 0.9664316217562359], ['w_1c6465a', 'w_fc10c15', 0.9167778846212059], ['w_090c801', 'w_0abdaf4', 1.0162480407036587], ['w_9d86594', 'w_4a9436a', 0.8438448829194773], ['w_6cfa650', 'w_f94039d', 0.791008774774737], ['w_10f67bb', 'w_955bfe2', 0.7255329601979759], ['w_e966f36', 'w_15805cd', 0.8998237161607208], ['w_9573686', 'w_3904956', 0.6865933518205334], ['w_d223b2e', 'w_f48451c', 0.9170670873175155], ['w_f3a12bc', 'w_b570fe1', 1.0213613534972965], ['w_1b2bf0f', 'w_00904a7', 0.8264174654194819], ['w_bbfce38', 'w_a47dc2d', 0.6120373771130394], ['w_0b398b2', 'w_1032bb6', 1.0471967431160114]]
results=[['w_c815410', 'w_32eac70', 1.0516533281733955], ['w_b9e5911', 'w_065ee1b', 0.8286123938472945], ['w_c073f96', 'w_bd9f354', 0.9205367729223889], ['w_b5e6c9c', 'w_0a0c768', 0.7817477741311869], ['w_33696ac', 'w_93fddcb', 0.8916725193097135], ['w_8431ae8', 'w_50fd107', 0.8034099887472066], ['w_5793f1c', 'w_fe8aee2', 0.7613702552051208], ['w_3815890', 'w_b938e96', 0.930676131205607], ['w_659bdb8', 'w_b495218', 0.8257822390639562], ['w_9a9a2c8', 'w_32de0c5', 0.9943817643695652], ['w_d980663', 'w_7b05eeb', 0.9090920971462331], ['w_a059841', 'w_b0cadc0', 0.9502690702134184], ['w_4c0a2c9', 'w_9472e22', 0.7165683691802086], ['w_eb13108', 'w_02469a1', 0.797317404599841], ['w_2e374c0', 'w_180e241', 0.7640425472374486], ['w_bfcad53', 'w_857cb95', 1.0830052683069062], ['w_025911c', 'w_b311fb6', 0.9913169903071029], ['w_ae6ac74', 'w_52a5d92', 0.9881278494483812], ['w_34d7623', 'w_ab44ae4', 0.9504095522694886], ['w_1c6465a', 'w_fc10c15', 0.9467712720981858], ['w_090c801', 'w_4bdd289', 0.9656182832365576], ['w_9d86594', 'w_4a9436a', 0.8587659025877432], ['w_d24c654', 'w_a763725', 0.9085028901309208], ['w_10f67bb', 'w_955bfe2', 0.7271121307514723], ['w_e966f36', 'w_15805cd', 0.9102271882648882], ['w_9573686', 'w_3904956', 0.6850377822227538], ['w_d223b2e', 'w_f48451c', 0.9191913694597358], ['w_f3a12bc', 'w_5a3e0de', 0.9253087075150188], ['w_1b2bf0f', 'w_00904a7', 0.8377499821772271], ['w_bbfce38', 'w_a47dc2d', 0.6177815891176082]]
for result in results:
    expect_class=result[0]
    predict_class=result[1]
    expect_class_path=train_classed_path+expect_class
    predict_class_path=train_classed_path+predict_class
    for image_name in class_name_2_image_names_all.get(expect_class):
        if image_name in test_names:
            test_file_path=test_dir+image_name
            test_file_name=image_name
    dst_sub_dir=dst_dir+expect_class
    if not os.path.exists(dst_sub_dir):
        os.makedirs(dst_sub_dir)
    for file_name in os.listdir(expect_class_path):
        file_path=os.path.join(expect_class_path,file_name)
        dst_file_path=os.path.join(dst_sub_dir,expect_class+'.'+file_name)
        shutil.copyfile(file_path, dst_file_path)
    for file_name in os.listdir(predict_class_path):
        file_path=os.path.join(predict_class_path,file_name)
        dst_file_path=os.path.join(dst_sub_dir,predict_class+'.'+file_name)
        shutil.copyfile(file_path, dst_file_path)
    shutil.copyfile(test_file_path,os.path.join(dst_sub_dir,'test.'+test_file_name))
        
        
        
        
    
import json
def concat_json(type='train'): #type-> val or train (default: train)
        print("-------",type,"--------")
        if type =='val':
                file = open(f'{type}/Bitter_rot_val.json')
        else:
                file = open(f'{type}/Bitter_rot.json')
        json_data = json.load(file)
        for target in ['scab', 'white_rot','sooty_blotch','flyspeck','Bitter pit','Brown rot']:
                if type =='val':
                        file2 = open(f'{type}/{target}_val.json')
                else:
                        file2 = open(f'{type}/{target}.json')
                json_data.update(json.load(file2))

        with open(f'{type}/via_region_data.json', 'w', encoding='utf-8') as make_file:
                json.dump(json_data, make_file)

        #--------check----------
        from collections import defaultdict
        file_check = open(f'{type}/via_region_data.json')
        json_data_ = json.load(file_check)
        print(len(json_data_.keys()))
        counter = defaultdict(int)
        for e,k in enumerate(json_data_.keys()):
                try:
                        for kk in json_data_[k]['regions'][0]['region_attributes']['apple'].keys():
                                counter[kk] += 1
                except: #형식에 맞지 않는 경우 error
                        print(k)
                        print(json_data_[k]['regions'])

        print(counter) #각 class 개수 print
concat_json()

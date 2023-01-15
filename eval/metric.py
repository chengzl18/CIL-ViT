import json
import csv
import math

# path to your evaluation logs
path = '_benchmarks_results/test_CoRL2017_Town02-xx'

def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def load_csv(file):
    with open(file, 'r', encoding='utf-8') as f:
        rows = [row for row in csv.DictReader(f, delimiter=',')]
    return rows


for weathers in [{'4'},{'14'},{'4','14'}]:
    print('\n'+str(weathers))
    lines = load_csv(path+'/summary.csv')
    total = 0
    success = 0
    success_time = 0
    for line in lines:
        if line['weather'] in weathers and line['exp_id']=='0':
            if line['result']=='1': 
                success+=1
                success_time+=float(line['final_time'])
            total+=1
    print('total:{}, success:{}, rate:{:.2f}%'.format(total, success, 100*success/total))
    print('average success time:{:.2f}s'.format(success_time/success if success else 0))


    lines = load_csv(path+'/measurements.csv')
    n_collision, n_intersection, n_fraction = 0, 0, 0
    current_id = ''
    collision, intersection = False, False
    
    acummulated_distance = 0.0
    prev_pos = None
    
    for line in lines:
        if line['weather'] in {'14'} and line['exp_id']=='0':
            start_point, end_point = line['start_point'], line['end_point']
            pos = (float(line['pos_x']), float(line['pos_y']))
            id = start_point+','+ end_point
            if id!=current_id:
                if collision:
                    n_collision += 1
                if intersection:
                    n_intersection += 1
                if collision or intersection:
                    n_fraction += 1
                current_id=id
                collision, intersection = False, False
                prev_pos = pos
            if (not collision) and (not intersection):
                acummulated_distance += sldist(pos, prev_pos)
                prev_pos = pos
            if float(line['collision_other'])>0 or float(line['collision_pedestrians'])>0 or float(line['collision_vehicles'])>0: 
                collision=True
            if float(line['intersection_otherlane'])>0 or float(line['intersection_offroad'])>0: 
                intersection=True

    if collision:
        n_collision += 1
    if intersection:
        n_intersection += 1

    print('total:{}, collision:{}, rate:{:.2f}%'.format(total, n_collision, 100*n_collision/total))
    print('total:{}, intersection:{}, rate:{:.2f}%'.format(total, n_intersection, 100*n_intersection/total))

    km = acummulated_distance/1000.0
    print('distance before fraction:{}, fraction:{}, rate:{:.2f}%'.format(km, n_fraction, km/n_fraction))


    
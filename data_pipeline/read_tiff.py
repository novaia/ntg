import os, argparse, glob

def main():
    os.environ['GDAL_PAM_ENABLED'] = 'NO'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    input_paths = glob.glob(f'{args.input_path}/*/*.tif')
    #input_paths = os.listdir(args.input_path)
    for path in input_paths:
        temp_tif = 'data/temp.tif'
        input_tif = os.path.join(args.input_path, path)
        fillnodata_cmd = f'gdal_fillnodata.py -of GTiff -md 100 {input_tif} {temp_tif}'
        os.system(fillnodata_cmd)
        
        output_png = os.path.join(args.output_path, f'{path[:-4]}_2.png')
        translate_cmd = f'gdal_translate -of PNG -scale {temp_tif} {output_png}'
        os.system(translate_cmd)

    os.remove(temp_tif)


if __name__ == '__main__':
    main()

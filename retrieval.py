import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from extract_feats import single_picture

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Retrieval')
    parser.add_argument('--query_path', default='./query/test.jpg', type=str,help='query image name')
    parser.add_argument('--data_base', default='./database/database.pth',type=str, help='queried database')
    parser.add_argument('--retrieval_num', default=5, type=int, help='retrieval number')

    opt = parser.parse_args()

    query_path, data_base, retrieval_num = opt.query_path, opt.data_base, opt.retrieval_num

    database = torch.load(data_base)
    query_image = Image.open(query_path).convert('RGB').resize((224, 224),resample=Image.BILINEAR)
    plt.imshow(query_image)
    plt.axis('off')
    plt.title('query_image')
    plt.show()
    query_feats = single_picture(query_path)
    gallery_dirs = database['img']

    gallery_feats = database['features']

    dist_matrix = torch.cdist(query_feats.unsqueeze(0).unsqueeze(0), gallery_feats.unsqueeze(0)).squeeze()

    index = dist_matrix.topk(k=retrieval_num, dim=-1, largest=False)[1]
    for i, idx in enumerate(index):
        retrieval_dist = dist_matrix[idx.item()].item()
        retrieval_image = Image.open(gallery_dirs[idx.item()]).convert('RGB').resize((224, 224),resample=Image.BILINEAR)
        plt.imshow(retrieval_image)
        plt.axis('off')
        plt.title('dis : {:.4f}'.format(retrieval_dist))
        plt.show()

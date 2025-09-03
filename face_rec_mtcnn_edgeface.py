#!/usr/bin/env python3
"""
MTCNN + EdgeFace pipeline
Requires: clone edgeface repo into EDGEREPO path and have checkpoint file available.
Commands same as FaceNet script but model/ckpt args required.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

# adjust this to your local clone
EDGEREPO = os.environ.get('EDGEREPO') or '/home/bacus/workspace/edgeface'
sys.path.append(EDGEREPO)

# import repo-specific functions
from backbones import get_model
from face_alignment import align

edgeface_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def l2_normalize_np(x, eps=1e-10):
    denom = np.sqrt(np.maximum(np.sum(x * x, axis=1, keepdims=True), eps))
    return x / denom


def load_edgeface(model_name, ckpt_path, device):
    model = get_model(model_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def load_mtcnn(device):
    return MTCNN(keep_all=True, device=device, image_size=112, margin=20, post_process=False)


def embed_faces_edgeface(img_rgb_np, boxes, model, device):
    if boxes is None or len(boxes) == 0:
        return None
    embs = []
    for (x1, y1, x2, y2) in boxes.astype(int):
        crop = img_rgb_np[y1:y2, x1:x2, :]
        if crop.size == 0:
            continue
        pil = Image.fromarray(crop).convert('RGB')
        tensor = edgeface_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor)
        embs.append(emb.cpu().numpy()[0])
    if not embs:
        return None
    embs = np.vstack(embs).astype(np.float32)
    embs = l2_normalize_np(embs)
    return embs


def cosine_sim_matrix(A, B):
    An = l2_normalize_np(A.copy())
    Bn = l2_normalize_np(B.copy())
    return An @ Bn.T


def match_embeddings(query_embs, bank_vectors, bank_names, thresh=0.6):
    sims = cosine_sim_matrix(query_embs, bank_vectors)
    idx = sims.argmax(axis=1)
    scores = sims[np.arange(sims.shape[0]), idx]
    labels = [bank_names[j] if scores[i] >= thresh else 'others' for i, j in enumerate(idx)]
    return labels, scores


def draw_boxes(img_bgr, boxes, labels, scores=None):
    for i, b in enumerate(boxes.astype(int)):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = labels[i]
        if scores is not None:
            text = f"{text} ({scores[i]:.2f})"
        tsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1 - tsize[1] - 6), (x1 + tsize[0] + 6, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img_bgr


# Facebank functions
def build_facebank(dataset_dir, out_path, model_name, ckpt, device, min_imgs=1):
    device = 'cuda' if torch.cuda.is_available() and device in (None, 'cuda') else 'cpu'
    mtcnn = load_mtcnn(device)
    model = load_edgeface(model_name, ckpt, device)
    names, vectors = [], []
    for person in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, person)
        if not os.path.isdir(pdir):
            continue
        emb_list = []
        for fn in sorted(os.listdir(pdir)):
            fpath = os.path.join(pdir, fn)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            boxes, _ = mtcnn.detect(pil)
            if boxes is None or len(boxes) == 0:
                continue
            box = boxes[np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))][None, :]
            embs = embed_faces_edgeface(img_rgb, box, model, device)
            if embs is None:
                continue
            emb_list.append(embs[0])
        if len(emb_list) >= min_imgs:
            mean_emb = l2_normalize_np(np.mean(np.vstack(emb_list), axis=0, keepdims=True))[0]
            names.append(person)
            vectors.append(mean_emb)
            print(f"[OK] {person}: {len(emb_list)} images")
        else:
            print(f"[SKIP] {person}: not enough faces")
    if not names:
        raise RuntimeError('No valid people found')
    np.savez(out_path, names=np.array(names, dtype=object), vectors=np.vstack(vectors).astype(np.float32))
    print('Saved facebank ->', out_path)
    return names


def load_facebank(path):
    data = np.load(path, allow_pickle=True)
    return data['names'].tolist(), data['vectors'].astype(np.float32)


# Inference
def infer_image(facebank_path, image_path, out_path, model_name, ckpt, device, thresh):
    device = 'cuda' if torch.cuda.is_available() and device in (None, 'cuda') else 'cpu'
    mtcnn = load_mtcnn(device)
    model = load_edgeface(model_name, ckpt, device)
    bank_names, bank_vectors = load_facebank(facebank_path)
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError('Cannot read image')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    boxes, _ = mtcnn.detect(pil)
    if boxes is not None and len(boxes) > 0:
        embs = embed_faces_edgeface(img_rgb, boxes, model, device)
        if embs is not None:
            labels, scores = match_embeddings(embs, bank_vectors, bank_names, thresh)
            img = draw_boxes(img, boxes, labels, scores)
    cv2.imwrite(out_path, img)
    print('Saved', out_path)


def infer_video(facebank_path, video_source, model_name, ckpt, device, thresh, out_path):
    device = 'cuda' if torch.cuda.is_available() and device in (None, 'cuda') else 'cpu'
    mtcnn = load_mtcnn(device)
    model = load_edgeface(model_name, ckpt, device)
    bank_names, bank_vectors = load_facebank(facebank_path)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video source')
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    headless = os.environ.get('DISPLAY') is None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        boxes, _ = mtcnn.detect(pil)
        if boxes is not None and len(boxes) > 0:
            embs = embed_faces_edgeface(img_rgb, boxes, model, device)
            if embs is not None:
                labels, scores = match_embeddings(embs, bank_vectors, bank_names, thresh)
                frame = draw_boxes(frame, boxes, labels, scores)
        if writer:
            writer.write(frame)
        if not headless:
            cv2.imshow('EdgeFace (MTCNN)', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    cap.release()
    if writer:
        writer.release()
    if not headless:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# CLI
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)

    b = sp.add_parser('build')
    b.add_argument('--dataset', required=True)
    b.add_argument('--out', default='./facebank_edge.npz')
    b.add_argument('--model', default='edgeface_s_gamma_05')
    b.add_argument('--ckpt', required=True)
    b.add_argument('--device', default=None)

    ii = sp.add_parser('infer-image')
    ii.add_argument('--facebank', required=True)
    ii.add_argument('--image', required=True)
    ii.add_argument('--out', default='./out_edge.jpg')
    ii.add_argument('--thresh', type=float, default=0.6)
    ii.add_argument('--model', default='edgeface_s_gamma_05')
    ii.add_argument('--ckpt', required=True)
    ii.add_argument('--device', default=None)

    iv = sp.add_parser('infer-video')
    iv.add_argument('--facebank', required=True)
    iv.add_argument('--video', required=True)
    iv.add_argument('--out', default=None)
    iv.add_argument('--thresh', type=float, default=0.6)
    iv.add_argument('--model', default='edgeface_s_gamma_05')
    iv.add_argument('--ckpt', required=True)
    iv.add_argument('--device', default=None)

    args = p.parse_args()
    if args.cmd == 'build':
        build_facebank(args.dataset, args.out, args.model, args.ckpt, args.device)
    elif args.cmd == 'infer-image':
        infer_image(args.facebank, args.image, args.out, args.model, args.ckpt, args.device, args.thresh)
    elif args.cmd == 'infer-video':
        source = 0 if args.video == '0' else args.video
        infer_video(args.facebank, source, args.model, args.ckpt, args.device, args.thresh, args.out)
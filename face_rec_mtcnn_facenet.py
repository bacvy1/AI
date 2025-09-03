#!/usr/bin/env python3
"""
MTCNN + FaceNet pipeline
Commands:
  build --dataset <dir> --out facebank.npz
  infer-image --facebank facebank.npz --image input.jpg --out out.jpg
  infer-video --facebank facebank.npz --video 0 --out out.avi

Notes:
- Facebank saved as .npz with keys: names, vectors
- Uses cosine similarity for matching
- Headless-safe (no imshow if DISPLAY not set)
"""

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def l2_normalize(x, eps=1e-10):
    return x / np.sqrt(np.maximum(np.sum(x * x, axis=1, keepdims=True), eps))


def load_models(device):
    mtcnn = MTCNN(keep_all=True, device=device, image_size=160, margin=20, post_process=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet


def detect_boxes(mtcnn, img_rgb):
    boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))
    if boxes is None:
        return np.empty((0, 4))
    h, w = img_rgb.shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes.astype(int)


def embed_faces_facenet(img_rgb, boxes, resnet, device):
    embs = []
    for (x1, y1, x2, y2) in boxes.astype(int):
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pil = Image.fromarray(crop).resize((160, 160)).convert('RGB')
        arr = np.asarray(pil).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.tensor(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(tensor)
        embs.append(emb.cpu().numpy()[0])
    if not embs:
        return None
    embs = np.vstack(embs).astype(np.float32)
    embs = l2_normalize(embs)
    return embs


def cosine_sim(A, B):
    A = l2_normalize(A.copy())
    B = l2_normalize(B.copy())
    return A @ B.T


def match_embeddings(query_embs, bank_vectors, bank_names, thresh=0.5):
    sims = cosine_sim(query_embs, bank_vectors)
    idx = sims.argmax(axis=1)
    scores = sims[np.arange(sims.shape[0]), idx]
    labels = [bank_names[i] if scores[k] >= thresh else 'others' for k, i in enumerate(idx)]
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


# Facebank build/load
def build_facebank(dataset_dir, out_path, device, min_imgs=1):
    mtcnn, resnet = load_models(device)
    names, vectors = [], []
    for person in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, person)
        if not os.path.isdir(pdir):
            continue
        emb_list = []
        for fn in sorted(os.listdir(pdir)):
            fpath = os.path.join(pdir, fn)
            try:
                img_bgr = cv2.imread(fpath)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                continue
            boxes = detect_boxes(mtcnn, img_rgb)
            if boxes.shape[0] == 0:
                continue
            embs = embed_faces_facenet(img_rgb, np.array([boxes[0]]), resnet, device)
            if embs is None:
                continue
            emb_list.append(embs[0])
        if len(emb_list) >= min_imgs:
            mean_emb = l2_normalize(np.mean(np.vstack(emb_list), axis=0, keepdims=True))[0]
            names.append(person)
            vectors.append(mean_emb)
            print(f"[OK] {person}: {len(emb_list)} images")
        else:
            print(f"[SKIP] {person}: not enough faces")
    if not names:
        raise RuntimeError('No valid people found to build facebank')
    np.savez(out_path, names=np.array(names, dtype=object), vectors=np.vstack(vectors).astype(np.float32))
    print('Saved facebank ->', out_path)
    return names


def load_facebank(path):
    data = np.load(path, allow_pickle=True)
    names = data['names'].tolist()
    vectors = data['vectors'].astype(np.float32)
    return names, vectors


# Inference
def infer_image(facebank_path, image_path, out_path, device, thresh):
    mtcnn, resnet = load_models(device)
    bank_names, bank_vectors = load_facebank(facebank_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError('Cannot read image')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_boxes(mtcnn, img_rgb)
    if boxes.shape[0] > 0:
        embs = embed_faces_facenet(img_rgb, boxes, resnet, device)
        labels, scores = match_embeddings(embs, bank_vectors, bank_names, thresh)
        img_bgr = draw_boxes(img_bgr, boxes, labels, scores)
    cv2.imwrite(out_path, img_bgr)
    print('Saved', out_path)


def infer_video(facebank_path, video_source, out_path, device, thresh):
    mtcnn, resnet = load_models(device)
    bank_names, bank_vectors = load_facebank(facebank_path)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video source')

    writer = None
    if out_path is not None:
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
        boxes = detect_boxes(mtcnn, img_rgb)
        if boxes.shape[0] > 0:
            embs = embed_faces_facenet(img_rgb, boxes, resnet, device)
            labels, scores = match_embeddings(embs, bank_vectors, bank_names, thresh)
            frame = draw_boxes(frame, boxes, labels, scores)
        if writer:
            writer.write(frame)
        if not headless:
            cv2.imshow('FaceRec (FaceNet)', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
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
    b.add_argument('--out', default='./facebank_facenet.npz')
    b.add_argument('--device', default=None)

    ii = sp.add_parser('infer-image')
    ii.add_argument('--facebank', required=True)
    ii.add_argument('--image', required=True)
    ii.add_argument('--out', default='./out_facenet.jpg')
    ii.add_argument('--thresh', type=float, default=0.6)
    ii.add_argument('--device', default=None)

    iv = sp.add_parser('infer-video')
    iv.add_argument('--facebank', required=True)
    iv.add_argument('--video', required=True)
    iv.add_argument('--out', default=None)
    iv.add_argument('--thresh', type=float, default=0.6)
    iv.add_argument('--device', default=None)

    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() and (args.device in (None, 'cuda')) else 'cpu'

    if args.cmd == 'build':
        build_facebank(args.dataset, args.out, device)
    elif args.cmd == 'infer-image':
        infer_image(args.facebank, args.image, args.out, device, args.thresh)
    elif args.cmd == 'infer-video':
        source = 0 if args.video == '0' else args.video
        infer_video(args.facebank, source, args.out, device, args.thresh)
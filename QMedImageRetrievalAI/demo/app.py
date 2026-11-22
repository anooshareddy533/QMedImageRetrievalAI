#!/usr/bin/env python3
"""Local-only Gradio prototype; research retrieval, not clinical software."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from qmed.training.trainer import load_checkpoint
from qmed.retrieval.faiss_index import RetrievalIndex
from qmed.utils.config import resolve_device
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from retrieve import embed_query


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--index-dir',required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=7860)
    args=p.parse_args()
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError('Install UI extra: pip install gradio>=4') from e
    device=resolve_device('auto')
    model,cfg,_=load_checkpoint(args.checkpoint,device)
    index=RetrievalIndex.load(args.index_dir)
    def search(image,modality,target,k):
        if not image:
            return {'error':'Upload an image to search.'}
        vec=embed_query(model,device,cfg['data']['image_size'],modality,image=image)
        matches=index.search(vec,k=int(k),target_modality=target if target!='All' else None)
        return {'image_only_query':True,'modality':modality,'top_k':matches,
                'caution':'UCCS is category-level research relevance, not patient matching or diagnostic evidence.'}
    with gr.Blocks(title='QMedImageRetrievalAI (research demo)') as ui:
        gr.Markdown('## QMedImageRetrievalAI\nLocal research prototype · Not a diagnostic or clinical decision-support tool.\nFor full four-sequence MRI queries, use `scripts/retrieve.py`.')
        with gr.Row():
            img=gr.Image(type='filepath',label='Image (CT / MRI / X-ray)')
            with gr.Column():
                mod=gr.Dropdown(['CT','MRI','XRAY'],value='CT',label='Query modality')
                target=gr.Dropdown(['All','CT','MRI','XRAY'],value='All',label='Target modality')
                k=gr.Slider(minimum=1,maximum=20,step=1,value=10,label='Top K')
                button=gr.Button('Retrieve similar images',variant='primary')
        output=gr.JSON(label='Ranked retrieval results')
        button.click(search,inputs=[img,mod,target,k],outputs=[output])
    ui.launch(server_name=args.host,server_port=args.port,share=False)

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    main()

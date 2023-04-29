nll_checkpoint=0.554715_6_736581.pth
python bin/ensemble.py config/RRG/baseline-mimic.yml \
    dataset.seq.processing=ifcc_clean_report \
    dataset.image.root=data/RRG/mimic-cxr/findings/ \
    dataset.seq.root=data/RRG/mimic-cxr/findings/ \
    dataset.seq.file=findings.tok \
    dataset.seq.tokenizer_max_len=128 \
    dataset.image.multi_image=3 \
    model.cnn.backbone=densenet121 \
    model.cnn.visual_embedding_dim=1024 \
    ensemblor.batch_size=16 \
    ensemblor.beam_width=2 \
    ensemblor.metrics='[chexbert,radgraph,ROUGEL]' \
    ensemblor.splits='[test]' \
    ckpt_dir=ckpt \
    ckpt=${nll_checkpoint} \
    name=nll_findings_bertscore_128
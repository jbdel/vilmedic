nll_checkpoint=0.554715_6_736581.pth

concept_type=all_concepts
python bin/ensemble.py config/RRG/baseline-mimic-force.yml \
    dataset.seq.processing=ifcc_clean_report \
    dataset.image.root=data/RRG/mimic-cxr/findings/ \
    dataset.seq.root=data/RRG/mimic-cxr/findings/ \
    dataset.seq.file=findings.tok \
    dataset.seq.tokenizer_max_len=128 \
    dataset.force_seq.processing=ifcc_clean_report \
    dataset.force_seq.root=data/RRG/mimic-cxr/concepts/ \
    dataset.force_seq.file=${concept_type}.tok \
    dataset.force_seq.tokenizer_max_len=128 \
    dataset.image.multi_image=3 \
    model.cnn.backbone=densenet121 \
    model.cnn.visual_embedding_dim=1024 \
    ensemblor.batch_size=1 \
    ensemblor.beam_width=2 \
    ensemblor.metrics='[chexbert,radgraph,ROUGEL]' \
    ensemblor.splits='[test]' \
    ckpt_dir=ckpt \
    ckpt=${nll_checkpoint} \
    name=nll_findings_constrain_gt_${concept_type}_bertscore_128

# concept_type=concat_concepts
# python bin/ensemble.py config/RRG/baseline-mimic-force.yml \
#     dataset.seq.processing=ifcc_clean_report \
#     dataset.image.root=data/RRG/mimic-cxr/findings/ \
#     dataset.seq.root=data/RRG/mimic-cxr/findings/ \
#     dataset.seq.file=findings.tok \
#     dataset.seq.tokenizer_max_len=128 \
#     dataset.force_seq.processing=ifcc_clean_report \
#     dataset.force_seq.root=data/RRG/mimic-cxr/concepts/ \
#     dataset.force_seq.file=${concept_type}.tok \
#     dataset.force_seq.tokenizer_max_len=128 \
#     dataset.image.multi_image=3 \
#     model.cnn.backbone=densenet121 \
#     model.cnn.visual_embedding_dim=1024 \
#     ensemblor.batch_size=1 \
#     ensemblor.beam_width=2 \
#     ensemblor.metrics='[chexbert,radgraph,ROUGEL]' \
#     ensemblor.splits='[test]' \
#     ckpt_dir=ckpt \
#     ckpt=${nll_checkpoint} \
#     name=nll_findings_constrain_gt_${concept_type}_bertscore_128
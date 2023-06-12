# rl_checkpoint=0.617898_11_358234.pth
N_TRAIN_CONSTRAINTS=0
NAME=rl_findings_scst_chexpert_concepts_gt_nconstraints${N_TRAIN_CONSTRAINTS}_bertscore_128
RL_CKPT=/home/cvanuden/git-repos/vilmedic/ckpt/${NAME}/0.110064_1_464619.pth
CONCEPT_TYPE=all_concepts

# python bin/ensemble.py config/RRG/baseline-mimic.yml \
#     dataset.seq.processing=ifcc_clean_report \
#     dataset.image.root=data/RRG/mimic-cxr/findings/ \
#     dataset.seq.root=data/RRG/mimic-cxr/findings/ \
#     dataset.seq.file=findings.tok \
#     dataset.seq.tokenizer_max_len=128 \
#     dataset.image.file=image.tok \
#     dataset.image.image_path=data/images/ \
#     dataset.image.multi_image=3 \
#     model.proto=RRG_SCST \
#     model.top_k=0 \
#     model.scores_weights='[0.01,0.495,0.495]' \
#     model.scores_args='[{},{reward_level: partial}]' \
#     model.scores=[bertscore,radgraph] \
#     model.use_nll=true \
#     model.cnn.backbone=densenet121 \
#     model.cnn.visual_embedding_dim=1024 \
#     ensemblor.batch_size=16 \
#     ensemblor.beam_width=2 \
#     ensemblor.metrics='[BLEU,ROUGEL,CIDERD,bertscore,radgraph,chexbert,radentitymatchexact,radentitynli]' \
#     ensemblor.splits=[validate,test] \
#     ensemblor.ckpt=${RL_CKPT} \
#     ckpt_dir=ckpt \
#     name=${NAME}



for N_CONSTRAINTS in "1" "-1"
do
    mkdir /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_scst_chexpert_concepts_gt_nconstraints${N_TRAIN_CONSTRAINTS}_${N_CONSTRAINTS}_bertscore_128
    cp ${RL_CKPT} /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_scst_chexpert_concepts_gt_nconstraints${N_TRAIN_CONSTRAINTS}_${N_CONSTRAINTS}_bertscore_128
    cp /home/cvanuden/git-repos/vilmedic/ckpt/${NAME}/vocab.tgt /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_scst_chexpert_concepts_gt_nconstraints${N_TRAIN_CONSTRAINTS}_${N_CONSTRAINTS}_bertscore_128

    python bin/ensemble.py config/RRG/baseline-mimic-force.yml \
        dataset.seq.processing=ifcc_clean_report \
        dataset.image.root=data/RRG/mimic-cxr/findings/ \
        dataset.image.file=subset.image.tok \
        dataset.seq.root=data/RRG/mimic-cxr/findings/ \
        dataset.seq.file=subset.findings.tok \
        dataset.seq.tokenizer_max_len=128 \
        dataset.force_seq.processing=ifcc_clean_report \
        dataset.force_seq.root=data/RRG/mimic-cxr/concepts/ \
        dataset.force_seq.file=subset.${CONCEPT_TYPE}.tok \
        dataset.force_seq.tokenizer_max_len=128 \
        dataset.force_seq.num_concepts=$N_CONSTRAINTS \
        dataset.image.multi_image=3 \
        model.proto=RRG_SCST \
        model.top_k=0 \
        model.scores_weights='[0.01,0.495,0.495]' \
        model.scores_args='[{},{reward_level: partial}]' \
        model.scores=[bertscore,radgraph] \
        model.use_nll=true \
        model.cnn.backbone=densenet121 \
        model.cnn.visual_embedding_dim=1024 \
        ensemblor.batch_size=1 \
        ensemblor.beam_width=2 \
        ensemblor.metrics='[BLEU,ROUGEL,CIDERD,bertscore,radgraph,chexbert,radentitymatchexact,radentitynli]' \
        ensemblor.splits='[test]' \
        ensemblor.prune_factor=200 \
        ensemblor.sat_tolerance=2 \
        ensemblor.alpha=0.05 \
        ensemblor.beta=0.25 \
        ensemblor.look_ahead_step=4 \
        ensemblor.look_ahead_width=2 \
        ensemblor.look_ahead_sample=False \
        ensemblor.fusion_t=1 \
        ensemblor.ckpt=${RL_CKPT} \
        ckpt_dir=ckpt \
        name=rl_findings_scst_chexpert_concepts_gt_nconstraints${N_TRAIN_CONSTRAINTS}_${N_CONSTRAINTS}_bertscore_128
done
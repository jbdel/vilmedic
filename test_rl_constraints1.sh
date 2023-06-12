RL_CKPT=0.617898_11_358234.pth
CONCEPT_TYPE=all_concepts
N_CONSTRAINTS=1

for LOOK_AHEAD_STEP in 4
do
    mkdir /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_${CONCEPT_TYPE}_constraints_gt_nconstraints${N_CONSTRAINTS}_lookahead${LOOK_AHEAD_STEP}_bertscore_128
    cp /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_all_concepts_constraints_gt_1_bertscore_128/0.617898_11_358234.pth /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_${CONCEPT_TYPE}_constraints_gt_nconstraints1_lookahead${LOOK_AHEAD_STEP}_bertscore_128
    cp /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_all_concepts_constraints_gt_1_bertscore_128/vocab.tgt /home/cvanuden/git-repos/vilmedic/ckpt/rl_findings_${CONCEPT_TYPE}_constraints_gt_nconstraints1_lookahead${LOOK_AHEAD_STEP}_bertscore_128
    
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
        ensemblor.look_ahead_step=$LOOK_AHEAD_STEP \
        ensemblor.look_ahead_width=2 \
        ensemblor.look_ahead_sample=False \
        ensemblor.fusion_t=1 \
        ensemblor.ckpt=${RL_CKPT} \
        ckpt_dir=ckpt \
        name=rl_findings_${CONCEPT_TYPE}_constraints_gt_nconstraints1_lookahead${LOOK_AHEAD_STEP}_bertscore_128
done
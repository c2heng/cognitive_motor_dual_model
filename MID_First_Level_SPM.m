Prep = '/fmriprep';
sub = dir(fullfile(Prep,'sub-*'));
sub = sub([sub.isdir]);
Output =[ '/' category];

fmri_t = 40;
fmri_t0 = 20;
fmri_rt = 2.2;
fmri_Units =  'secs';

first_sub = zeros(1,length(sub));
parfor i = 1:length(sub)
    disp(i)
    func = dir(fullfile(Prep,sub(i).name,ses,'func','Preprocessed*task-mid*.nii'));
    if ~isempty(func)
        first_sub(i) = i;
    end
end
first_sub(first_sub == 0) = [];
sub = sub(first_sub);
%%
wrong_subject = zeros(1,length(sub));
sub_job = cell(1,length(sub));
for j = 1:length(sub)
    try
        disp(['Subject    ',num2str(j,'%012d')]);
        func = dir(fullfile(Prep,sub(j).name,ses,'func','Preprocessed*task-mid*.nii'));
        motion = dir(fullfile(Prep,sub(j).name,ses,'func','MID_Covas.txt'));
        
        onset_files = fullfile(Prep,sub(j).name,ses,'func',onset_file_name);

        clear matlabbatch
        matlabbatch{1}.spm.stats.fmri_spec.mthresh=-inf;
        matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(fullfile(Output,ses,sub(j).name));
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = fmri_Units;
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = fmri_rt;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = fmri_t;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = fmri_t0;

        % nii file
        niifile = spm_select('ExtFPList',func.folder,func.name);
        matlabbatch{1}.spm.stats.fmri_spec.sess.scans = cellstr(niifile);
        % onset
        sub_onset = load(fullfile(onset_files.folder,onset_files.name));
        conditions = table2struct(sub_onset.Onset_mat);
        for k = 1:length(conditions)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(k).name = char(conditions(k).Condition_name);
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(k).tmod = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(k).pmod = struct('name', {}, 'param', {}, 'poly', {});
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(k).onset    = conditions(k).Onset;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(k).duration = conditions(k).Duration';
        end
        
        % Nuisance Regressors: motion
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
        motionfiles = spm_select('FPList',motion.folder,[motion.name,'.*']);
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = cellstr(motionfiles);
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
    
        % keyboard(Defaults,need't change)
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0;
        matlabbatch{1}.spm.stats.fmri_spec.mask = {'/MNI152_T1_3mm_brain_mask.nii'};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
        % model estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep;
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tname = 'Select SPM.mat';
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(1).name = 'filter';
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(1).value = 'mat';
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(2).name = 'strtype';
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(2).value = 'e';
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).sname = 'fMRI model specification: SPM.mat File';
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).src_exbranch = substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1});
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1).src_output = substruct('.','spmmat');
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
        % SPM contrasts
        matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep;
        matlabbatch{3}.spm.stats.con.spmmat(1).tname = 'Select SPM.mat';
        matlabbatch{3}.spm.stats.con.spmmat(1).tgt_spec{1}(1).name = 'filter';
        matlabbatch{3}.spm.stats.con.spmmat(1).tgt_spec{1}(1).value = 'mat';
        matlabbatch{3}.spm.stats.con.spmmat(1).tgt_spec{1}(2).name = 'strtype';
        matlabbatch{3}.spm.stats.con.spmmat(1).tgt_spec{1}(2).value = 'e';
        matlabbatch{3}.spm.stats.con.spmmat(1).sname = 'Model estimation: SPM.mat File';
        matlabbatch{3}.spm.stats.con.spmmat(1).src_exbranch = substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1});
        matlabbatch{3}.spm.stats.con.spmmat(1).src_output = substruct('.','spmmat');
        % SPM contrast set
        for k = 1:length(conditions)
            cond_vector_run = zeros(1,length(conditions)+26);
            cond_vector_run(k)= 1;
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.name = char(conditions(k).Condition_name);
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.convec = cond_vector_run;
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.sessrep = 'none';
        end       
        
        sub_job{j}=  matlabbatch; 
    catch
        wrong_subject(j) = j;
    end
end



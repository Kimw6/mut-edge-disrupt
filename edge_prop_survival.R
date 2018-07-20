source('common.R')
source('summary.coxph.R')
library(survival)
library(glmnet)
library(rhdf5)

script_label = 'edge_prop_survival'
output.dir = create.output.dir.func(script_label)
data.dir = create.data.dir.func(script_label)

survival_dir = find.newest.data.path('tcga_xml_to_survival_csv')
survival_data_all = read.csv(ospj(survival_dir, 'survival.csv'), row.names=1)
#row.names(survival_data_all) = survival_data_all$patient_id

load_prop_data = function() {
  edge_prop_input_dir = find.newest.data.path('propagate_mutations_edges_alpha_0.80')
  edge_prop_data_path = ospj(edge_prop_input_dir, 'data_propagated.hdf5')

  edge_prop_raw = h5read(edge_prop_data_path, '/mutations')
  edge_prop = data.frame(t(edge_prop_raw$block0_values), row.names=edge_prop_raw$axis1)
  names(edge_prop) = edge_prop_raw$axis0

  edge_prop
}

edge_prop_all = load_prop_data()

edge_indexes = grep('_', names(edge_prop_all))
gene_indexes = grep('_', names(edge_prop_all), invert=TRUE)

patients = Reduce(
  intersect,
  list(
    row.names(edge_prop_all),
    row.names(survival_data_all)
  )
)

surv = Surv(survival_data_all[patients, 'survival_time'], survival_data_all[patients, 'dead'])
s = data.frame(
  surv=surv,
  row.names=patients
)

edge_prop = edge_prop_all[patients,]

surv_perm_count = 1000
edge_count = 100
rsq_surv_results = rep(0, surv_perm_count)
pvalue_surv_results = rep(0, surv_perm_count)

edge_prop_var = sort(apply(edge_prop[,edge_indexes], 2, var), decreasing=TRUE)
top_edges = names(edge_prop_var[1:edge_count])
remaining_edges = names(edge_prop_var[edge_count:length(edge_prop_var)])

top_surv_df = data.frame(surv, edge_prop[,top_edges])
cc = coxph.control(iter.max=500)
top_edge_cph = coxph(surv ~ ., data=top_surv_df, control=cc)
top_edge_cph_summary = summary.coxph.custom(top_edge_cph)
top_rsq = top_edge_cph_summary$rsq[1]
top_pvalue = top_edge_cph_summary$logtest['pvalue']

for (i in 1:surv_perm_count) {
  cat(paste(c('Processing iteration ', i, '\n'), collapse=''))
  edge_selection = sample(remaining_edges, edge_count)
  random_surv_df = data.frame(surv, edge_prop[,edge_selection])
  cc = coxph.control(iter.max=100)
  random_edge_cph = coxph(surv ~ ., data=random_surv_df, control=cc)
  random_edge_cph_summary = summary.coxph.custom(random_edge_cph)
  rsq_surv_results[i] = random_edge_cph_summary$rsq[1]
  pvalue_surv_results[i] = random_edge_cph_summary$logtest['pvalue']
}

save.fig(output.dir(append.extension('surv_rsq_hist')))
hist(
  rsq_surv_results,
  breaks=50,
  main=expression(R^2 ~ 'from random edges vs. top edges'),
  xlab=expression(R^2)
)
abline(v=top_rsq, col='#FF0000FF', lwd=3)
dev.off()

nl10_pvalue_surv_results = -log10(pvalue_surv_results)
save.fig(output.dir(append.extension('surv_nl10_pvalue_hist')))
hist(
  nl10_pvalue_surv_results,
  breaks=50,
  main='-log10(P-value) from random edges vs. top edges',
  xlab='-log10(P-value)'
)
abline(v=(-log10(top_pvalue)), col='#FF0000FF', lwd=3)
dev.off()

hist(rsq_surv_results, breaks=50)

# Univariate survival results

surv_result_cols = c('coef', 'pvalue', 'concordance', 'r_square', 'max_r_square')
univariate_surv_results = data.frame(
  matrix(0, ncol=length(surv_result_cols), nrow=dim(edge_prop)[2])
)
names(univariate_surv_results) = surv_result_cols
row.names(univariate_surv_results) = dimnames(edge_prop)[[2]]

for (i in 1:dim(edge_prop)[2]) {
  tryCatch(
    {
      prop_vec = edge_prop[,i]
      cph = coxph(surv ~ prop_vec)
      cph_summary = summary.coxph.custom(cph)
      univariate_surv_results[i, 'coef'] = cph$coefficients
      univariate_surv_results[i, 'pvalue'] = cph_summary$logtest['pvalue']
      univariate_surv_results[i, 'concordance'] = cph_summary$concordance[1]
      univariate_surv_results[i, 'r_square'] = cph_summary$rsq[1]
      univariate_surv_results[i, 'max_r_square'] = cph_summary$rsq[2]
    },
    error=function(e) {e}
  )
}
univariate_surv_results[,'minus_log10_pvalue'] = -log10(univariate_surv_results[,'pvalue'])

save(univariate_surv_results, file=data.dir('univariate_surv_results.RData'))
write.csv(univariate_surv_results, file=data.dir('univariate_surv_results.csv'))

edge_surv_results = univariate_surv_results[edge_indexes,]
gene_surv_results = univariate_surv_results[gene_indexes,]

rsq_edge_surv_result_order = order(edge_surv_results$r_square, decreasing=TRUE)
rsq_edge_surv_results = edge_surv_results[rsq_edge_surv_result_order[1:100], 'r_square']

rsq_gene_surv_result_order = order(gene_surv_results$r_square, decreasing=TRUE)
rsq_gene_surv_results = gene_surv_results[rsq_gene_surv_result_order[1:100], 'r_square']

save.fig(output.dir(append.extension('surv-rsquare-cdfs')))
plot(
  ecdf(rsq_gene_surv_results),
  do.points=FALSE,
  verticals=TRUE,
  col=matplotlib_colors[1],
  main=expression(R^2 ~ 'CDF: genes vs. edges'),
  xlab=expression(R^2),
  ylab='CDF'
)
plot(
  ecdf(rsq_edge_surv_results),
  do.points=FALSE,
  verticals=TRUE,
  col=matplotlib_colors[2],
  add=TRUE
)
legend(
  'topleft',
  legend=c('Genes', 'Edges'),
  lty=1,
  col=matplotlib_colors[1:2]
)
ks_result = ks.test(rsq_edge_surv_results, rsq_gene_surv_results)
text_pvalue(ks_result$p.value)
dev.off()

nz_sel = surv[,1] > 0
sv_train_nz = s[nz_sel, 'surv']

edge_prop_nz = edge_prop[nz_sel,]

surv_model = glmnet(
  as.matrix(edge_prop_nz),
  as.matrix(sv_train_nz),
  family='cox',
  alpha=1
)
coefs = coef(surv_model)
coef_nz = abs(coefs) > 0
coef_usage = rowSums(coef_nz)
print(coef_usage[coef_usage > 0])

save.fig(output.dir(append.extension('coefs')))
plot(surv_model)
dev.off()

save(coefs, coef_nz, coef_usage, file=data.dir('l1_cox_coefs.RData'))

# /Univariate survival results


# after this be dragons

descs = c('mut_only', 'edge_only', 'both')
enet_models = list()
pvalues = data.frame(
  matrix(NA, ncol=length(descs), nrow=k)
)
names(pvalues) = descs

lambda_cutoff = 0.02

k = 5
kfl = kfold_indexes(length(patients), k)

for (i in 1:k) {
  cat(paste(c('Processing iteration ', i, '\n'), collapse=''))
  patients_train = patients[kfl[[i]]$train]
  patients_test = patients[kfl[[i]]$test]

  edge_prop_train = edge_prop[patients_train,]
  edge_prop_test = edge_prop[patients_test,]

  edge_prop_train = edge_prop[patients_train,]
  edge_prop_test = edge_prop[patients_test,]

  sv_train = s[patients_train, 'surv']
  sv_test = s[patients_test, 'surv']

  nz_sel = sv_train[,1] > 0
  sv_train_nz = sv_train[nz_sel]

  # Both
  tryCatch(
    {
      edge_prop_train_nz = edge_prop_train[nz_sel,]
      enet_train = glmnet(as.matrix(edge_prop_train_nz), as.matrix(sv_train_nz), family='cox', alpha=1, maxit=1000000)
      usable_lambdas = which(enet_train$lambda <= lambda_cutoff)
      if (length(usable_lambdas) > 0) {
        enet_first_lambda_leq_cutoff = usable_lambdas[1]
      } else {
        enet_first_lambda_leq_cutoff = length(enet_train$lambda)
      }

      enet_predictions_test = predict(
        enet_train,
        newx=as.matrix(edge_prop_test),
        type='response'
      )[,enet_first_lambda_leq_cutoff]

      enet_sf = survfit(sv_test ~ enet_predictions_test)
      enet_pvalue = summary.coxph.custom(coxph(sv_test ~ enet_predictions_test))$logtest['pvalue']
      pvalues[i, 'both'] = enet_pvalue
    },
    error=function(e) e
  )

  # Mutations only
  tryCatch(
    {
      edge_prop_train_nz = edge_prop_train[nz_sel, gene_indexes]
      enet_train = glmnet(as.matrix(edge_prop_train_nz), as.matrix(sv_train_nz), family='cox', alpha=1)
      usable_lambdas = which(enet_train$lambda <= lambda_cutoff)
      if (length(usable_lambdas) > 0) {
        enet_first_lambda_leq_cutoff = usable_lambdas[1]
      } else {
        enet_first_lambda_leq_cutoff = length(enet_train$lambda)
      }

      enet_predictions_test = predict(
        enet_train,
        newx=as.matrix(edge_prop_test[,gene_indexes]),
        type='response'
      )[,enet_first_lambda_leq_cutoff]

      enet_sf = survfit(sv_test ~ enet_predictions_test)
      enet_pvalue = summary.coxph.custom(coxph(sv_test ~ enet_predictions_test))$logtest['pvalue']
      pvalues[i, 'mut_only'] = enet_pvalue
    },
    error=function(e) e
  )

  # Edges only
  tryCatch(
    {
      edge_prop_train_nz = edge_prop_train[nz_sel, edge_indexes]
      enet_train = glmnet(as.matrix(edge_prop_train_nz), as.matrix(sv_train_nz), family='cox', alpha=1)
      usable_lambdas = which(enet_train$lambda <= lambda_cutoff)
      if (length(usable_lambdas) > 0) {
        enet_first_lambda_leq_cutoff = usable_lambdas[1]
      } else {
        enet_first_lambda_leq_cutoff = length(enet_train$lambda)
      }

      enet_predictions_test = predict(
        enet_train,
        newx=as.matrix(edge_prop_test[,edge_indexes]),
        type='response'
      )[,enet_first_lambda_leq_cutoff]

      enet_sf = survfit(sv_test ~ enet_predictions_test)
      enet_pvalue = summary.coxph.custom(coxph(sv_test ~ enet_predictions_test))$logtest['pvalue']
      pvalues[i, 'edge_only'] = enet_pvalue
    },
    error=function(e) e
  )
}

nl10_pvalues = -log10(pvalues)

save.fig(append.extension(output.dir('pvalue_boxplot')))
boxplot(
  nl10_pvalues,
  col=matplotlib_colors,
  main='L1 Cox Regression 5-Fold Cross-Validation P-Values',
  ylab='-log10(P-value)'
)
dev.off()

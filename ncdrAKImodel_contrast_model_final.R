library(xgboost)
library(doMC)
library(caret)
library(onehot)
library(mgcv)


source("prediction_performance.R")
source("prediction_performance_calib.R")

################################################################
# generate pre-procedural AKI model
################################################################
# load data on which the preprocedural model was built
load("PreProcVars_10052011.RData")
vars=preprocvars1
data0=final.imp[,vars]
vars.ex=c("allMVSupport","PCICardioShock","Shock",
          "Height","Weight","PrePCILVEFcat","egfrc","Anemia")
data.preproc=data0[,-which(vars %in% vars.ex)] # 947,091

# build model
sel=c("egfr","PreProcHgb","Prior2weeksHFcat=1","PriorCardioShock","Age",
      "PCIStatus=1","PCIStatus=3","PrePCILVEF","Diabetescat=3","Diabetescat=1",
      "BMI","CADPresentationcat=4","Prior2weeksHFcat=5","PriorHF",
      "PriorCardiacArrest" ,"AdmtSource=1")
ind.out=33
y=data.preproc[,ind.out]
x=predict(onehot(data.preproc[,-ind.out]),data.preproc[,-ind.out])
x=x[,colnames(x)%in%sel]
ntrees=1000
set.seed(1127)
tmp=xgb.cv(data=x,label=as.numeric(y), nfold=5, nrounds=ntrees, objective="binary:logistic",
           early_stopping_rounds = 50, metrics=c("auc"),
           eta=0.1, subsample=0.9, colsample_bytree=0.6, max_depth=6, 
           min_child_weight=3, print_every_n = 10,
           stratified = TRUE, nthread=32,  seed=1127)
newnrounds=tmp$best_iteration
set.seed(1130)
model.preproc1=xgboost(data=x,label=as.numeric(y),nrounds=newnrounds, objective="binary:logistic",
                       eta=0.1, max_depth=6, min_child_weight=3, subsample=0.9,
                       colsample_bytree=0.6, eval_metric=c("auc"), print_every_n = 10
)
################################################################

################################################################
# construct predictors for the contrast model
################################################################
# preprocedural AKI risk estimate
# load new data
load("cohort_byyear_d07142017.RData")
ind.out=1
tmp=data.new.preproc.byyear
timeframe=data.new.preproc.byyear$timeframe
x.new=predict(onehot(tmp[,-ind.out]),tmp[,-ind.out])
x.new=x.new[,colnames(x.new)%in%sel]
y.new=tmp$AKI
# predictions on new data
pred.new.preproc=predict(model.preproc1,newdata=x.new)# prob estimate
pred.new.preproc.lp=predict(model.preproc1,newdata=x.new,outputmargin = T)# logits
# prediction.performance(pred.new.preproc,y.new,10)

# organize data and apply exclusions, imputation
load("excluded_VisitPepsdPsubm_d07142017.RData")
preCr=final.data$PreProcCreat
postCr=final.data$PostProcCreat
ctrvol=final.data$ContrastVol
newdialysis=final.data$PostDialysis==1
newdialysis[is.na(newdialysis)]=FALSE
inhospdeath=final.data$DCStatus==2
inhospdeath[is.na(inhospdeath)]=FALSE
procDate=final.data$ProcedureDate
dcDate=final.data$DCDate
otherMajSurg=final.data$OtherMajorSurgery==1
otherMajSurg[is.na(otherMajSurg)]=FALSE
lenofstay=dcDate-procDate
mydata=data.frame(prerisk=pred.new.preproc,prerisklp=pred.new.preproc.lp,preCr=preCr,postCr=postCr,
                  ctrvol=ctrvol,tf=timeframe,newdialysis=newdialysis,inhospdeath=inhospdeath,
                  lenofstay=lenofstay, othermajsurg=otherMajSurg,id=final.data$VisitKey, AKIold=y.new) #
# missing creatinines
mydata.ex=mydata
cex=which(is.na(mydata.ex$preCr)|is.na(mydata.ex$postCr))#
mydata.ex=mydata.ex[-cex,]
# missing contrast volumes
cex=which(is.na(mydata.ex$ctrvol))# 
mydata.ex=mydata.ex[-cex,]# 
# exclude baseline creatinine <0.3 or >4
cex=which(mydata.ex$preCr<=0.3|mydata.ex$preCr>4)# 
mydata.ex=mydata.ex[-cex,] # 
################################################################

################################################################
# construct outcome
################################################################
changeCr=mydata.ex$postCr-mydata.ex$preCr
AKIcat=rep(1,length(changeCr))
AKIcat[changeCr>=0.3]=2
AKIcat[changeCr>=0.5]=3
AKIcat[changeCr>=1.0]=4
mydata.ex$AKIcat=AKIcat
################################################################

################################################################
# development and temporal validation sets
################################################################
dev=mydata.ex[mydata.ex$tf%in%c(1:4),]# 
# temporal validation data
val=mydata.ex[mydata.ex$tf%in%c(5:6),]# 
# stats
dim(dev)
dim(val)
for (i in 1:3){
  print(length(which(dev$AKIcat>=(i+1))))
  print(mean(dev$AKIcat>=(i+1)))
  print(length(which(val$AKIcat>=(i+1))))
  print(mean(val$AKIcat>=(i+1)))
}
# save to csv for visualization in matlab
write.csv(dev,"deveopment_set.csv")
write.csv(val,"validation_set.csv")
################################################################

################################################################
# split development set into training and test
################################################################
set.seed(8172018)
c.split = createDataPartition(dev$AKIcat, p = 0.5, list = F)
train=dev[c.split,]
test=dev[-c.split,]
# stats
dim(train)
dim(test)
for (i in 1:3){
  print(length(which(train$AKIcat>=(i+1))))
  print(mean(train$AKIcat>=(i+1)))
  print(length(which(test$AKIcat>=(i+1))))
  print(mean(test$AKIcat>=(i+1)))
}
write.csv(test,'train_set.csv')
write.csv(test,'test_set.csv')
################################################################

################################################################
# Build contrast model
# multinomial logit link: levels0-3
# predictors: preprocedural AKI risk logit, contrast volume
# tensorproduct basis functions
################################################################
train$AKIcatm=train$AKIcat-1
fit.AKIcat5.m=gam(list(AKIcatm~te(ctrvol,prerisklp),~te(ctrvol,prerisklp),~te(ctrvol,prerisklp)),
                  method="REML",gamma=log(dim(train)[1])/5,family=multinom(K=3),data=train,qr=T)
################################################################

################################################################
# model selection
################################################################
# logistic regression with linear predictors with no interaction term
fit.AKIcat5.lr=gam(list(AKIcatm~ctrvol+prerisklp,~ctrvol+prerisklp,~ctrvol+prerisklp),
                   method="REML",gamma=log(dim(train)[1])/5,family=multinom(K=3),data=train,qr=T)
# logistic regression with linear predictors and interaction term
fit.AKIcat5.lri=gam(list(AKIcatm~ctrvol*prerisklp,~ctrvol*prerisklp,~ctrvol*prerisklp),
                    method="REML",gamma=log(dim(train)[1])/5,family=multinom(K=3),data=train,qr=T)
# with nonlinear predictors and no interaction term
fit.AKIcat5.tis=gam(list(AKIcatm~s(ctrvol)+s(prerisklp),~s(ctrvol)+s(prerisklp),~s(ctrvol)+s(prerisklp)),
                    method="REML",gamma=log(dim(train)[1])/5,family=multinom(K=3),data=train,qr=T)
# with nonlinear predictors and interaction terms via tensorproducts
fit.AKIcat5.tes=gam(list(AKIcatm~s(ctrvol)+s(prerisklp)+ti(ctrvol,prerisklp),~s(ctrvol)+s(prerisklp)+ti(ctrvol,prerisklp),~s(ctrvol)+s(prerisklp)+ti(ctrvol,prerisklp)),
                    method="REML",gamma=log(dim(train)[1])/5,family=multinom(K=3),data=train,qr=T)

# anova for nested models
anova(fit.AKIcat5.lr,fit.AKIcat5.lri,test="Chisq")
anova(fit.AKIcat5.tis,fit.AKIcat5.tes,test="Chisq")

# AIC
AIC(fit.AKIcat5.lr,fit.AKIcat5.lri,fit.AKIcat5.tis,fit.AKIcat5.m)
# BIC
BIC(fit.AKIcat5.lr,fit.AKIcat5.lri,fit.AKIcat5.tis,fit.AKIcat5.m)
################################################################

################################################################
# performance evaluation on test set
################################################################
pred.test.AKIcat5m=predict(fit.AKIcat5.m,newdata=test,type="response")
# performance on test set
pred.test=pred.test.AKIcat5m
prediction.performance(pred.test[,2]+pred.test[,3]+pred.test[,4],test$AKIcat>=2,10)
prediction.performance(pred.test[,3]+pred.test[,4],test$AKIcat>=3,10)
prediction.performance(pred.test[,4],test$AKIcat>=4,10)
# Calibration plots
# traditional plot: by deciles
par(mfrow=c(1,3))
par(pty="s")
out=prediction.performance.calib(pred.test[,2]+pred.test[,3]+pred.test[,4],test$AKIcat>=2,10)
x=seq(0,0.3,by=0.01)
pred=predict(out[[1]],newdata=data.frame(pred=x),se.fit=T)
plot(out[[7]],out[[6]],xlab="Predicted Rate",ylab="Observed Rate",xlim=c(0,0.3),ylim=c(0,0.3))
lines(c(0,1),c(0,1),xlab="",ylab="",type="l",col="red")
out=prediction.performance.calib(pred.test[,3]+pred.test[,4],test$AKIcat>=3,10)
plot(out[[7]],out[[6]],xlab="Predicted Rate",ylab="Observed Rate",xlim=c(0,0.3),ylim=c(0,0.3))
pred=predict(out[[1]],newdata=data.frame(pred=x),se.fit=T)
lines(x,pred$fit)
lines(c(0,1),c(0,1),xlab="",ylab="",type="l",col="red")
out=prediction.performance.calib(pred.test[,4],test$AKIcat>=4,10)
plot(out[[7]],out[[6]],xlab="Predicted Rate",ylab="Observed Rate",xlim=c(0,0.3),ylim=c(0,0.3))
pred=predict(out[[1]],newdata=data.frame(pred=x),se.fit=T)
lines(x,pred$fit)
lines(c(0,1),c(0,1),xlab="",ylab="",type="l",col="red")
# smooth plot
myd=data.frame(x2=pred.test[,2]+pred.test[,3]+pred.test[,4],x3=pred.test[,3]+pred.test[,4],
               x4=pred.test[,4],
               y=test$AKIcat)
fit.calib2=gam((y>=2)~s(x2,bs="bs"),method="REML",data=myd,qr=T)
vcov.calib=vcovHC(fit.calib2,type="HC0")
fit.calib2$Vp=vcov.calib
fit.calib3=gam((y>=3)~s(x3,bs="bs"),method="REML",data=myd,qr=T)
vcov.calib=vcovHC(fit.calib3,type="HC0")
fit.calib3$Vp=vcov.calib
fit.calib4=gam((y>=4)~s(x4,bs="bs"),method="REML",data=myd,qr=T)
vcov.calib=vcovHC(fit.calib4,type="HC0")
fit.calib4$Vp=vcov.calib
par(mfrow=c(1,3))
x=seq(0,1,by=0.001)
par(pty="s")
pred.calib2=predict(fit.calib2,newdata=data.frame(x2=x),se.fit=T)
plot(x,pred.calib2$fit,type="l",xlim=c(0,1),ylim=c(0,1),xlab="Predicted Rate",ylab="Observed Rate")
lines(x,pred.calib2$fit-1.96*pred.calib2$se.fit,lty=2)
lines(x,pred.calib2$fit+1.96*pred.calib2$se.fit,lty=2)
lines(c(0,1),c(0,1),col="red")
pred.calib3=predict(fit.calib3,newdata=data.frame(x3=x),se.fit=T)
plot(x,pred.calib3$fit,type="l",xlim=c(0,1),ylim=c(0,1),xlab="Predicted Rate",ylab="Observed Rate")
lines(x,pred.calib3$fit-1.96*pred.calib3$se.fit,lty=2)
lines(x,pred.calib3$fit+1.96*pred.calib3$se.fit,lty=2)
lines(c(0,1),c(0,1),col="red")
pred.calib4=predict(fit.calib4,newdata=data.frame(x4=x),se.fit=T)
plot(x,pred.calib4$fit,type="l",xlim=c(0,1),ylim=c(0,1),xlab="Predicted Rate",ylab="Observed Rate")
lines(x,pred.calib4$fit-1.96*pred.calib4$se.fit,lty=2)
lines(x,pred.calib4$fit+1.96*pred.calib4$se.fit,lty=2)
lines(c(0,1),c(0,1),col="red")
################################################################


################################################################
# performance evaluation on validation set
################################################################
pred.val=predict(fit.AKIcat5.m,newdata=val,type="response")
prediction.performance(pred.val[,2]+pred.val[,3]+pred.val[,4],val$AKIcat>=2,10)
prediction.performance(pred.val[,3]+pred.val[,4],val$AKIcat>=3,10)
prediction.performance(pred.val[,4],val$AKIcat>=4,10)
# calibration plots
# by deciles
par(mfrow=c(1,3))
par(pty="s")
out=prediction.performance.calib(pred.val[,2]+pred.val[,3]+pred.val[,4],val$AKIcat>=2,10)
x=seq(0,0.3,by=0.01)
pred=predict(out[[1]],newdata=data.frame(pred=x),se.fit=T)
plot(out[[7]],out[[6]],xlab="Predicted Rate",ylab="Observed Rate",xlim=c(0,0.3),ylim=c(0,0.3))
lines(x,pred$fit)
lines(c(0,1),c(0,1),xlab="",ylab="",type="l",col="red")
out=prediction.performance.calib(pred.val[,3]+pred.val[,4],val$AKIcat>=3,10)
plot(out[[7]],out[[6]],xlab="Predicted Rate",ylab="Observed Rate",xlim=c(0,0.3),ylim=c(0,0.3))
pred=predict(out[[1]],newdata=data.frame(pred=x),se.fit=T)
lines(x,pred$fit)
lines(c(0,1),c(0,1),xlab="",ylab="",type="l",col="red")
out=prediction.performance.calib(pred.val[,4],val$AKIcat>=4,10)
plot(out[[7]],out[[6]],xlab="Predicted Rate",ylab="Observed Rate",xlim=c(0,0.3),ylim=c(0,0.3))
pred=predict(out[[1]],newdata=data.frame(pred=x),se.fit=T)
lines(x,pred$fit)
lines(c(0,1),c(0,1),xlab="",ylab="",type="l",col="red")
# smoothers
myd=data.frame(x2=pred.val[,2]+pred.val[,3]+pred.val[,4],x3=pred.val[,3]+pred.val[,4],
               x4=pred.val[,4],
               y=val$AKIcat)
fit.calib2=gam((y>=2)~s(x2,bs="cr"),method="REML",data=myd,qr=T)
vcov.calib=vcovHC(fit.calib2,type="HC0")
fit.calib2$Vp=vcov.calib
fit.calib3=gam((y>=3)~s(x3,bs="cr"),method="REML",data=myd,qr=T)
vcov.calib=vcovHC(fit.calib3,type="HC0")
fit.calib3$Vp=vcov.calib
fit.calib4=gam((y>=4)~s(x4,bs="cr"),method="REML",data=myd,qr=T)
vcov.calib=vcovHC(fit.calib4,type="HC0")
fit.calib4$Vp=vcov.calib
par(mfrow=c(1,3))
x=seq(0,1,by=0.001)
par(pty="s")
pred.calib2=predict(fit.calib2,newdata=data.frame(x2=x),se.fit=T)
plot(x,pred.calib2$fit,type="l",xlim=c(0,1),ylim=c(0,1),xlab="Predicted Rate",ylab="Observed Rate")
lines(x,pred.calib2$fit-1.96*pred.calib2$se.fit,lty=2)
lines(x,pred.calib2$fit+1.96*pred.calib2$se.fit,lty=2)
lines(c(0,1),c(0,1),col="red")
pred.calib3=predict(fit.calib3,newdata=data.frame(x3=x),se.fit=T)
plot(x,pred.calib3$fit,type="l",xlim=c(0,1),ylim=c(0,1),xlab="Predicted Rate",ylab="Observed Rate")
lines(x,pred.calib3$fit-1.96*pred.calib3$se.fit,lty=2)
lines(x,pred.calib3$fit+1.96*pred.calib3$se.fit,lty=2)
lines(c(0,1),c(0,1),col="red")
pred.calib4=predict(fit.calib4,newdata=data.frame(x4=x),se.fit=T)
plot(x,pred.calib4$fit,type="l",xlim=c(0,1),ylim=c(0,1),xlab="Predicted Rate",ylab="Observed Rate")
lines(x,pred.calib4$fit-1.96*pred.calib4$se.fit,lty=2)
lines(x,pred.calib4$fit+1.96*pred.calib4$se.fit,lty=2)
lines(c(0,1),c(0,1),col="red")
################################################################


################################################################
# visualization
################################################################
# surface contour plot
ctrvol.samp=seq(0,1000,by=1)
x=ctrvol.samp
n_x=length(x)
y=c(0.0001,seq(0.001,1-0.001,by=0.001),1-0.0001)
ylp=log(y/(1-y))
n_y=length(y)
pd=data.frame(ctrvol=rep(x,n_x),prerisk=rep(y,each=n_y),prerisklp=rep(ylp,each=n_y))
prediction.AKIcat5m.surf=predict(fit.AKIcat5.m,newdata = pd,type = "response")
write.csv(prediction.AKIcat5m.surf,"AKIcat5m_meshgrid.csv")

# plot as function of contrast volume
ctrvol.samp=seq(0,1000,by=1)
prerisk2ex=c(0.02,0.05,0.10,0.20,0.45,0.7,0.80,0.85)
prerisk2lex=log(prerisk2ex/(1-prerisk2ex))
nex=length(prerisk2lex)
n=length(ctrvol.samp)
newdata=data.frame(ctrvol=rep(ctrvol.samp,nex),
                   prerisk=rep(prerisk2ex,each=n),
                   prerisklp=rep(prerisk2lex,each=n),
                   PreProc_AKI=factor(rep(1:nex,each=n),levels=1:nex,labels=c(1:nex)))
pred.newdata=predict.gam(fit.AKIcat5.m,newdata,type="response")
mat.newdata=predict.gam(fit.AKIcat5.m,type="lpmatrix",newdata = newdata)
mat.newdata=mat.newdata[,1:25]
# plot trends in R
newdata$akirisk=pred.newdata[,2]+pred.newdata[,3]+pred.newdata[,4]
ggplot(newdata, aes(ctrvol,akirisk))+
  geom_line(aes(colour=PreProc_AKI),size=2)+
  labs(x="Contrast volume",y="Predicted AKI risk")+
  coord_cartesian(ylim=c(0,1))
# calculate confidence interval of predictions
# by bootstrap covariance of model parameters
set.seed(1203)
rmvn <- function(n,mu,sig) { ## MVN random deviates
  L <- mroot(sig);m <- ncol(L);
  t(mu + L%*%matrix(rnorm(m*n),m,n))
}
nboot=1e4
betaboot=rmvn(nboot,coef(fit.AKIcat5.m),fit.AKIcat5.m$Vp)
res.aki2=matrix(0,nboot,dim(newdata)[1])
res.aki3=res.aki2
res.aki4=res.aki2

for (i in 1:nboot){
  p2=exp(betaboot[i,1:25]%*%t(mat.newdata))
  p3=exp(betaboot[i,26:50]%*%t(mat.newdata))
  p4=exp(betaboot[i,51:75]%*%t(mat.newdata))
  prob=(p2+p3+p4)/(1+p2+p3+p4)
  res.aki2[i,]=log(prob/(1-prob))
  prob=(p3+p4)/(1+p2+p3+p4)
  res.aki3[i,]=log(prob/(1-prob))
  prob=(p4)/(1+p2+p3+p4)
  res.aki4[i,]=log(prob/(1-prob))
}
pred.aki2=pred.newdata[,2]+pred.newdata[,3]+pred.newdata[,4]
pred.aki2.logit=log(pred.aki2/(1-pred.aki2))
pred.aki3=pred.newdata[,3]+pred.newdata[,4]
pred.aki3.logit=log(pred.aki3/(1-pred.aki3))
pred.aki4=pred.newdata[,4]
pred.aki4.logit=log(pred.aki4/(1-pred.aki4))
mean.aki=pred.aki2.logit
sd.aki=apply(res.aki2,2,sd)  
maki2=pred.aki2
laki2=exp(mean.aki-1.96*sd.aki)/(1+exp(mean.aki-1.96*sd.aki))
haki2=exp(mean.aki+1.96*sd.aki)/(1+exp(mean.aki+1.96*sd.aki))
mean.aki=pred.aki3.logit
sd.aki=apply(res.aki3,2,sd)  
maki3=pred.aki3
laki3=exp(mean.aki-1.96*sd.aki)/(1+exp(mean.aki-1.96*sd.aki))
haki3=exp(mean.aki+1.96*sd.aki)/(1+exp(mean.aki+1.96*sd.aki))
mean.aki=pred.aki4.logit
sd.aki=apply(res.aki4,2,sd)  
maki4=pred.aki4
laki4=exp(mean.aki-1.96*sd.aki)/(1+exp(mean.aki-1.96*sd.aki))
haki4=exp(mean.aki+1.96*sd.aki)/(1+exp(mean.aki+1.96*sd.aki))
# save to csv for visualization in matlab
newdata$akirisk=maki2
newdata$llrisk=laki2
newdata$ulrisk=haki2
write.csv(newdata,"prediction_examples_aki2.csv")
newdata$akirisk=maki3
newdata$llrisk=laki3
newdata$ulrisk=haki3
write.csv(newdata,"prediction_examples_aki3.csv")
newdata$akirisk=maki4
newdata$llrisk=laki4
newdata$ulrisk=haki4
write.csv(newdata,"prediction_examples_aki4.csv")
################################################################

%% visualization 
%% scatter plot
% import development set
% generate KNN
K=500;
dataS=[ctrvol';prerisk'];
[idx1S,d1S]=knnsearch(dataS',dataS','K',K,...
'Distance','mahalanobis');
% change>=0.3
out=double(AKIcat>=2);
% change>=0.5
out=double(AKIcat>=3);
% change>=1.0
out=double(AKIcat>=4);
% generate observed rates
% 250-NN
scoreS=zeros(length(ctrvol),1);
parfor i=1:length(scoreS)
    scoreS(i)=nanmean(out(idx2S(i,1:250)));
end
figure, 
scatter(ctrvol,prerisk,15,scoreS,'filled')
h=colorbar;
caxis([0,0.7])
xlabel('Contrast volume (mL)')
ylabel('Pre-procedural AKI risk')
ylabel(h,'Observed rate')
colormap jet


%% create data mask based on density
x=0:1:1000;
y=[0.0001,0.001:0.001:(1-0.001),(1-0.0001)];
[X,Y]=meshgrid(y,x);
zmask=zeros(1001,1001);
rad2=50;
rad1=0.05;
for i=1:1001
    disp(num2str(i))
    parfor j=1:1001
        tmp1=X(i,j);
        tmp2=Y(i,j);
        c2=find(ctrvol>tmp2-rad2&ctrvol<tmp2+rad2);
        c1=find(prerisk>tmp1-rad1&prerisk<tmp1+rad1);
        c=intersect(c1,c2);
        zmask(i,j)=length(c);
    end
end
%% surface contour plot
% import surface data

% contour plot
z=V2+V3+V4;
z=V3+V4;
z=V4;
% generate plot
Z=reshape(z,[1001,1001]);
% restrict to area with at least 10-NN
z1=double(zmask>=10);
z1(z1==0)=NaN;
figure,pcolor(Y,X,Z.*z1);
hold on
shading interp
hc=contour(Y,X,Z,[0:0.1:0.6],'ShowText','on','LineColor','k');
h=colorbar;
caxis([0,0.7])
xlabel('Contrast volume (mL)')
ylabel('Pre-procedural AKI risk')
ylabel(h,'Predicted creatinine increase \geq 0.3 mg/dL')
colormap jet
axis([0,1000,0,1])

%% plot risk as function of contrast
zx=0:1:1000;
zy=[0.0001,0.001:0.001:(1-0.001),(1-0.0001)];
tmp1ex=[0.02,0.05,0.1,0.2,0.45,0.70,0.80,0.85];
n=length(tmp1ex);
fit=reshape(akirisk,[1001,n]);
ll=reshape(llrisk,[1001,n]);
ul=reshape(ulrisk,[1001,n]);
x0=0:1000;
figure,hold on
cmap=jet(n);

rad2=50;
rad1=0.05;
hlz=[];
for i=1:n
    y0=[ll(:,i)';fit(:,i)';ul(:,i)'];
    tmp1=tmp1ex(i);
    % find closest value in zmask
    [~,c1]=min(abs(tmp1-zy));
    c2=find(zmask(:,c1)>=10);
    select=c2;
    unsel=find(zmask(:,c1)<10);
    % plot those with at least 10-NN
    y=y0(:,select);
    x=x0(select);
    yn=y0(:,unsel);
    xn=x0(:,unsel); 
    px=[x,fliplr(x)];
    py=[y(1,:),fliplr(y(3,:))];
    patch(px,py,1,'FaceColor',cmap(i,:),'EdgeColor','none','FaceAlpha',.4)
    hl=plot(x,y(2,:),'Color',cmap(i,:),'LineWidth',2.5);
    hlz=[hlz,hl];
end
ylim([min(fit(:)),1])
xlabel('Contrast volume (mL)')
ylabel('Risk of creatine increase \geq 0.3 mg/dL')
legends={'2%','5%','10%','20%','45%','70%','80%','85%'};
hh=legend(hlz,legends);
set(get(hh,'title'),'String',{'Pre-procedural';'AKI risk'})
set(hh,'location','eastoutside')
xlim([0,1000])
set(gca,'xtick',[0:100:1000])
set(gca,'ytick',[0:0.1:1])
grid on
set(gca,'layer','top')
pbaspect([1.2 1 1])




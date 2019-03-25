clc;
clear all;
close all;
data=xlsread('Example.xlsx');      
[p,q]=size(data);                  
x=1;                                
y=1;
z=1;
for i=1:p
    if data(i,q)==0
        label1(x,1:q)=data(i,1:q);      
        x=x+1;
    end
    if data(i,q)==1
        label2(y,1:q)=data(i,1:q);      
        y=y+1;
    end
    if data(i,q)==2
        label3(z,1:q)=data(i,1:q);     
        z=z+1;
    end
end
mu1=mean(label1);           
mu2=mean(label2);
mu3=mean(label3);
mu=[mu1;mu2;mu3];          
mu_t=mean(data(:,[1,2]));

seperator=zeros(p,2);

seperator1=mu1(:,1:2)-mu_t;
seperator2=mu2(:,1:2)-mu_t;
seperator3=mu3(:,1:2)-mu_t;
seperator=[seperator1;seperator2;seperator3]
for i=1:2
    sb1(i)=seperator(:,i)'*seperator(:,i);
end
seperator_between=zeros(2,2);
for i=1:3
    seperator_between=seperator_between+seperator(i,:)'*seperator(i,:);
end
%%within
with=zeros(p,2);
for i=1:5
    for j=1:2
        with(i,j)=data(i,j)-mu1(j);
    end
end
for i=6:10
    for j=1:2
        with(i,j)=data(i,j)-mu2(j);
    end
end
for i=11:15
    for j=1:2
        with(i,j)=data(i,j)-mu3(j);
    end
end
seperator_within=zeros(2,2);
for i=1:p
    sw1=with(i,:)'*with(i,:);
    seperator_within=seperator_within+sw1;
end

mat=inv(seperator_within)*seperator_between;
[evec,eval] = eig(mat);
eval;
evec;
ld1=evec(:,1)'
Emat=ld1';
Final_matrix=data(:,1:2)*Emat;
y(1:15,1)=zeros();
plot(Final_matrix(1:5,1),y(1:5,1),'r*','linewidth',1.3);hold on;
plot(Final_matrix(6:10,1),y(6:10,1),'mo','linewidth',1.3);hold on;
plot(Final_matrix(11:15,1),y(11:15,1),'bs','linewidth',1.3);hold off;
legend('Label-1','Label-2','Label-3');
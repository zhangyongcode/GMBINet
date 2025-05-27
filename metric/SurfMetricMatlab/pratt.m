function [F,fac,msc] = pratt(Ea,Ed)

% Function EDPM : Edge detector performance measure function. 
% 	          Calculates for a given edge image the false alarm 
%		  count, miss count and figure of merit (F) values.   		 	
%
%  
%    Input(s)... Ea : Actual edge image     	
%		 Ed : Detected edge image.
%
%    Output(s).. fac: False alarm count
%		 msc: miss count
%		 F  : Figure of merit
Ea = im2double(Ea);
Ed = im2double(Ed);

% Ea=double(Ea);
% Ed=double(Ed);

[N,M] = size(Ea);
if [N,M] ~= size(Ed) 
  error('Actual and detected edge image sizes must be same');
end
a=0.1; % edge shift penalty constant;
fac=length(find((Ea-Ed)==-1)); % False Alarm Count
msc=length(find((Ea-Ed)==1));  % Miss Count
Na=sum(sum(Ea));Nd=sum(sum(Ed));
c=1/max(Na,Nd);
[ia,ja]=find(Ea==1);
for l=1:Na
  Aes(l)=Ed(ia(l),ja(l));
end
mi=ia(Aes==0);
mj=ja(Aes==0);
% mi=ia(find(Aes==0));
% mj=ja(find(Aes==0));
F=c*sum(Aes);
for k=1:length(mi) 
  n1=0;n2=0;m1=0;m2=0;
  while sum(sum(Ed(mi(k)-n1:mi(k)+n2,mj(k)-m1:mj(k)+m2)))<1
    if (1==(mi(k)-n1)) && (N==(mi(k)+n2)) && (1==(mj(k)-m1)) && (M==(mj(k)+m2))
        break;
    end
    if mi(k)-n1>1
        n1=n1+1;
    end
    if mi(k)+n2<N
        n2=n2+1;
    end  
    if mj(k)-m1>1
        m1=m1+1;
    end  
    if mj(k)+m2<M
        m2=m2+1;
    end
  end
  di=max([n1 n2 m1 m2]);
  F=F+c/(1+a*di^2);
end

F = F*100;

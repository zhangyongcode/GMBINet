function [Ea_gt, Ed_sal] = SalToEdge(GT, Sal) 
%  
%    Input(s)... Ea : Actual edge image     	
%		 Ed : Detected edge image.
%
%    Output(s).. fac: False alarm count
%		 msc: miss count
%		 F  : Figure of merit
    % ground truth
    GT = (GT > 128);
    GT = double(GT);
    [gy, gx] = gradient(GT);
    temp_edge = gy.*gy + gx.*gx;
    temp_edge(temp_edge~=0)=1;
    Ea_gt = uint8(temp_edge*255);    % gt
    % detect saliency map
    Sal = (Sal > 128);
    Sal = double( Sal);
    [sy, sx] = gradient( Sal);
    s_edge = sy.*sy + sx.*sx;
    s_edge(s_edge~=0)=1; 
    Ed_sal = uint8(s_edge*255);       % sal
    
end






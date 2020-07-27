manual_r = zeros(744,38);
manual_r(:,1) = manual{1,1}.';
for i = 2:38
    temp = manual{1,i};
    manual_r(:,i) = temp.';
end
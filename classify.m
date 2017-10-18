function class = classify(x, w, classes)
    result = w*(x');
    [M,I] = max(result);
    if M > 0
        class = classes(I);
    else
        class = NaN;
    end
end
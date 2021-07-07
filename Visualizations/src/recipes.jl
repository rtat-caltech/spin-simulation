@userplot BlochQuivers
@recipe function f(bq::BlochQuivers)
    frame, xaxis, yaxis = bq.args
    meanvec = unit(mean(frame, dims=2))[:,1]
end

@userplot BlochScatter
@recipe function f(bs::BlochScatter)
    frame, xaxis, yaxis, axlim = bs.args
    x, y = project(frame, xaxis, yaxis)
    seriestype --> :scatter
    aspect_ratio --> 1
    xlims --> (-axlim, axlim)
    ylims --> (-axlim, axlim)
    x .- mean(x), y .- mean(y)
end

@userplot ShadowPlot
@recipe function f(sp::ShadowPlot)
    frame = sp.args[1]
    meanvec = unit(mean(frame, dims=2))[:,1]
    meanx = meanvec[1]
    meany = meanvec[2]
    meanz = meanvec[3]

    aspect_ratio --> 1
    xlims --> (-1, 1)
    ylims --> (-1, 1)
    zlims --> (-1, 1)

    @series begin
        [0, meanx], [0, meany], [0, meanz]
    end

    # Draw the three shadows
    linealpha := 0.5
    primary := false
    @series begin
        [0, meanx], [0, meany], [-1, -1]
    end
    @series begin
        [-1, -1], [0, meany], [0, meanz]
    end
    @series begin
        [0, meanx], [1, 1], [0, meanz]
    end
end

@userplot BlochPlot
@recipe function f(bp::BlochPlot)
    frame, xaxis, yaxis, axlim = bp.args
    layout := @layout [scatter sphere]
            
    @series begin
        subplot --> 1
        BlochScatter((frame, xaxis, yaxis, axlim))
    end

    @series begin
        subplot --> 2
        ShadowPlot((frame,))
    end 
end


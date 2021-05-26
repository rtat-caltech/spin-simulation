@userplot BlochQuivers
@recipe function f(bq::BlochQuivers)
    frame, xaxis, yaxis = bq.args
    meanvec = unit(mean(frame, dims=2))[:,1]
end

@userplot BlochPlot
@recipe function f(bp::BlochPlot)
    frame, xaxis, yaxis, axlim = bp.args
    layout := @layout [scatter sphere]
    x, y = project(frame, xaxis, yaxis)
    
    meanvec = unit(mean(frame, dims=2))[:,1]
    meanx = meanvec[1]
    meany = meanvec[2]
    meanz = meanvec[3]
    
    shadowalpha = 0.3
    vectorcolor = :blue
    
    @series begin
        subplot --> 1
        seriestype --> :scatter
        aspect_ratio --> 1
        xlims --> (-axlim, axlim)
        ylims --> (-axlim, axlim)
        label --> "Spins"
        x .- mean(x), y .- mean(y)
    end
    
#     @series begin
#         subplot --> 2
#         seriestype --> :wireframe
#         linecolor --> :blue
#         sx, sy, sz
#     end
    
    xlims --> (-1, 1)
    ylims --> (-1, 1)
    zlims --> (-1, 1)
    subplot --> 2
    linecolor --> vectorcolor
    
    @series begin
        [0, meanx], [0, meany], [0, meanz]
    end
    
    linealpha --> shadowalpha
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

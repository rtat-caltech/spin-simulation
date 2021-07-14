const gravity= 9.81 #m^2/s
const zheight= 0.102 #m 
const xwidth = 0.4
const ylength= 0.076
const diffuse=0.01
const vmax=5;
using StaticArrays
using LinearAlgebra: dot, normalize
const SV = SVector{3,Float64} # We're working in 3D space here, folks.

abstract type AbstractParticle end

mutable struct FreeFallParticle <: AbstractParticle
    pos::SV
    vel::SV
    oldpos::SV
    oldvel::SV
    vmax::Float64
    time::Float64
    twall::Float64
    oldtwall::Float64
    whatwall::Int
    oldwhatwall::Int
    seed::Int
    oldseed::Int
end

function creatStructure()
    p = FreeFallParticle([0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], 0.0,0.0,0.0,0.0,1,1,1,1)

    return p;
end

function getIsotropic!(floor,ceiling,p::FreeFallParticle)
    #returns height of particle from Golub's UCN book, distribution from equation 4.7. (althought this is not 100% correct due to parabolic arcs)
    #stores the v^2 distributed from maximum velocity. 
    #not sure if this really works with the sphere, but its probably pretty good.
    vmax=4.0;
    L = ceiling-floor
    #hard v^2 distribution. at the lowest point in the geometry (ergodicity assumed)
	
    vmag=vmax*cbrt(rand(Float64))
	
    if vmag*vmag>2.0*gravity*L
    	vfactor=sqrt(1.0-L*gravity/vmag/vmag)
    	#height= 0.5/gravity*(vmag*vmag-cbrt(( (vfactor*((vmag)^3-L*gravity*vmag) + (vmag)^3 )^2*rand(Float64) + (vmag)^3)))   
        rn=rand(Float64);
        height=(vmag ^ 2 - (-rn * (L * gravity - vmag ^ 2) * sqrt(-L * gravity + vmag ^ 2) - vmag ^ 3 * (rn - 1)) ^ (2//3)) / gravity
    else
    	height=0.5/gravity*vmag*vmag*(1-cbrt((rand(Float64))^2))
    end

    height = clamp(height, 0.0, L)
     
    #magnitude of the ::SV juliavelocity at the height  
    #(it is possible that this populates velocities in places that the neutron cannot replicate from diffuse wall collisions alone, but it should be good enough. )
    vmagh=sqrt(vmag*vmag-2.0*gravity*height)
    thetap=acos(2.0*rand(Float64)-1.);
    phip= 2.0*pi*rand(Float64);
    #velocity components adjusted for height. 
    p.vel=SV(vmagh*sin(thetap)*cos(phip),vmagh*sin(thetap)*sin(phip),vmagh*cos(thetap));
    #return in geometry coordinates
    return height+floor;
end


function startInside!(p::FreeFallParticle)
    p.time=0.0;
   
    positionx=xwidth*(rand(Float64)-0.5);
    
    positiony=ylength*(rand(Float64)-0.5);
    
    positionz=getIsotropic!(-0.5*zheight,0.5*zheight,p);
    
    p.oldpos=SV(positionx,positiony,positionz);
        
    p.pos=p.oldpos;

    p.oldvel=p.vel;
    
    nextBoundary!(p)
end


function nextBoundary!(p::FreeFallParticle)
    positionx, positiony, positionz = p.pos
    velocityx, velocityy, velocityz = p.vel

    #this may look similar to the z walls in the cylinder, but its different.
    #+-zwall=z0+vz*twall-1/2*g*twall^2
    #why so much harder in julia?
    if velocityz*velocityz+2.0*(positionz-zheight/2.0)*gravity > 0
       
        twalltop=(velocityz-sqrt(velocityz*velocityz+2.0*(positionz-zheight/2.0)*gravity))/gravity;
       
        if velocityz*velocityz+2.0*(positionz+zheight/2.0)*gravity > 0
            
            twallbott=(velocityz+sqrt(velocityz*velocityz+2.0*(positionz+zheight/2.0)*gravity))/gravity;
            
            if twalltop <0.0
                twalltemp2=twallbott
            elseif twallbott<0.0
                twalltemp2=twalltop
            elseif twallbott<twalltop
                twalltemp2=twallbott
            else
                twalltemp2=twalltop
            end
        else
            twalltemp2=twalltop;
        end
    else
        twalltemp2=(velocityz+sqrt(velocityz*velocityz+2.0*(positionz+zheight/2.0)*gravity))/gravity;
    end
            

    twalltemp0=(xwidth/2.0-sign(velocityx)*positionx)/(sign(velocityx)*velocityx);
    twalltemp1=(ylength/2.0-sign(velocityy)*positiony)/(sign(velocityy)*velocityy);
   
    #goofball logic.
    if twalltemp2<=twalltemp1
        if twalltemp0<twalltemp2
            whatwall=0
        else
            whatwall=2
        end
    else
        if twalltemp0<twalltemp1
            whatwall=0
        else
            whatwall=1
        end
    end
    
    #whatwall=twalltemp2<=twalltemp1 ? (twalltemp0<twalltemp2 ? 0:2):(twalltemp0<twalltemp1 ? 0:1);
   
    if whatwall==0 
        twall=p.time+twalltemp0
    elseif whatwall==1
        twall=p.time+twalltemp1
    elseif whatwall==2
        twall=p.time+twalltemp2
    else
        twall=p.time
    end
    p.twall=twall;
    p.whatwall=whatwall;
    return twall
   
end

function moveParticle!(dt,p::FreeFallParticle)
 
    if p.time+dt<p.twall
        p.pos = p.pos .+ (p.vel .* dt)
        p.vel = p.vel .+ SV(0, 0, -gravity*dt)
        p.time=p.time+dt;
        #recursive until no wall is hit. 
        return; #the actual exit is here at the beginning
    end

    dtwall=p.twall-p.time;

    positionx=p.pos[1]+p.vel[1]*dtwall;
    positiony=p.pos[2]+p.vel[2]*dtwall;
    positionz=p.pos[3]+p.vel[3]*dtwall-0.5*gravity*dtwall*dtwall;
	
    velocityx=p.vel[1];
    velocityy=p.vel[2];
    velocityz=p.vel[2]-gravity*dtwall;
    whatwall=p.whatwall;
    twall=p.twall;

    #what wall is a c++ private class variable (how do you do this fast in Julia?, lots of overhead?)
    if whatwall==2
    
	#might be necessary to clamp to the wall if it hits the corner. 
   	#position[2]=position[2]>0? length/2.0:-length/2.0;
   			
	if rand(Float64)>diffuse
	    #specular
	    velocityz=-velocityz;
        
	else
	    #diffuse bounce on the ends. whats the cdf of cos(theta)^2*sin(theta) again? theta0=arccos((1-rand)^(1/3))
	    #ugg someone from boost should write a diffuse ditrbituion code, (they do have an arcsine dist) this is slow. 
	    #maybe template this in a spline for extra speed? lets see how bad it is. 
	    #method one.
            vmag=sqrt(velocityz*velocityz+velocityx*velocityx+velocityy*velocityy)
            temptheta=acos(cbrt(rand(Float64)))
            tempphi=2.0*pi*rand(Float64)
	    velocityz=-vmag*sign(positionz)*cos(temptheta)
	    velocityx=vmag*sin(temptheta)*cos(tempphi)
	    velocityy=vmag*sin(temptheta)*sin(tempphi)	  
        end
    elseif whatwall==1
	
	#might be necessary to clamp to the wall if it hits the corner. 
   	#position[2]=position[2]>0? length/2.0:-length/2.0;
   	
	if rand(Float64)>diffuse
	    #specular
	    
	    velocityy=-velocityy;
	    
	else
	    #diffuse bounce on the ends. whats the cdf of cos(theta)^2*sin(theta) again? theta0=arccos((1-rand)^(1/3))
	    #ugg someone from boost should write a diffuse ditrbituion code, (they do have an arcsine dist) this is slow. 
	    #maybe template this in a spline for extra speed? lets see how bad it is. 
	    #method one.
	    vmag=sqrt(velocityz*velocityz+velocityx*velocityx+velocityy*velocityy)
	    temptheta=acos(cbrt(rand(Float64)))
	    tempphi=2.0*pi*rand(Float64)
	    velocityy=-vmag*sign(positiony)*cos(temptheta)
	    velocityx=vmag*sin(temptheta)*cos(tempphi)
	    velocityz=vmag*sin(temptheta)*sin(tempphi)
	    
	    
        end
    else
	
	#might be necessary to clamp to the wall if it hits the corner. 
   	#position[2]=position[2]>0? length/2.0:-length/2.0;
   	
	if rand(Float64)>diffuse
	    #specular
	    
	    velocityx=-velocityx;
	    
	else
	    #diffuse bounce on the ends. whats the cdf of cos(theta)^2*sin(theta) again? theta0=arccos((1-rand)^(1/3))
	    #ugg someone from boost should write a diffuse ditrbituion code, (they do have an arcsine dist) this is slow. 
	    #maybe template this in a spline for extra speed? lets see how bad it is. 
	    
	    vmag=sqrt(velocityz*velocityz+velocityx*velocityx+velocityy*velocityy)
	    temptheta=acos(cbrt(rand(Float64)))
	    tempphi=2.0*pi*rand(Float64)
	    velocityx=-vmag*sign(positionx)*cos(temptheta)
	    velocityy=vmag*sin(temptheta)*cos(tempphi)
	    velocityz=vmag*sin(temptheta)*sin(tempphi)
	    
        end
   end

    #should have new velocity[] and position 
    #need to update time
    
    p.time=twall
    #maybe we will hit the wall again before dt is over? lets go to infinity and find out!
    
    p.pos=SV(positionx,positiony,positionz)
    p.vel=SV(velocityx,velocityy,velocityz)
    nextBoundary!(p)
    moveParticle!(dt-dtwall,p)
    
    return#exiting from recursive calls.
end

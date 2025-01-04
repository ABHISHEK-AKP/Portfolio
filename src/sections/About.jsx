import React, { useState } from 'react'
import Globe from 'react-globe.gl'
import Button from '../components/Button'

const About = () => {
    const [hasCopied, setHasCopied] = useState(false);
    const handleCopy = () => {
        navigator.clipboard.writeText('abhishekkumarpandey2803@gmail.com');
        setHasCopied(true);
        setTimeout(() => {
            setHasCopied(false);
        },2000 )
    }
  return (
    <section className='c-space my-20' id='about'>
        <div className='grid xl:grid-cols-3 xl:grid-rows-6 md:grid-cols-2 grid-cols-1 gap-5 h-full'>
            <div className='col-span-1 xl:row-span-3'>
                <div className='grid-container '>
                    <img src='/assets/grid1.png' alt="grid-1" className='w-full sm:h-[276px] h-fit object-contain'/>
                    <div>
                        <p className='grid-headtext'>Hi, I am Abhishek</p>
                        <p className='grid-subtext'>With 3 years of experience, I have honed my skills in front-end, back-end development, and AI projects. I am passionate about creating visually stunning and user-friendly interfaces that bring ideas to life.</p>
                    </div>
                </div>
            </div>
            <div className='col-span-1 xl:row-span-3'>
                 <div className='grid-container '>
                    <img src='/assets/grid2.png' alt="grid-2" className='w-full sm:h-[276px] h-fit object-contain'/>
                    <div>
                        <p className='grid-headtext'>TechStack</p>
                        <p className='grid-subtext'>I have expertise in React, Next.js, Node.js, Express.js, MongoDB, and AI technologies. I also have experience in Python, PHP, and Java.</p>
                    </div>
                 </div>
            </div>
            <div className='col-span-1 xl:row-span-4'>
                <div className='grid-container '>
                    <div className='rounded-3xl w-full sm:h-[326px] h-fit flex justify-center items-center'>
                        <Globe 
                        height={326} 
                        width={326} 
                        backgroundColor="rgba(0,0,0,0)" 
                        backGroundImageOpacity={0.5}
                        showAtmosphere
                        showGraticules 
                        globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
                        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
                        labelsData={[{
                            lat:51,
                            lng:85,
                            text:"I'm here",
                            color:"white",
                            size:20,
                        }]}
                        />
                    </div>
                    <div>
                        <p className='grid-headtext'>Location</p>
                        <p className='grid-subtext'>I am currently based in Ontario, Canada with remote work availability. I am open to new opportunities and collaborations in the tech industry.</p>
                        <Button name="Contact Me" isBeam containerClass="w-full mt-10"></Button>
                    </div>
                </div>
            </div>
            <div className='xl:col-span-2 xl:row-span-3'>
                <div className='grid-container '>
                    <img src='/assets/grid3.png' alt="grid-3" className='w-full sm-h-[266px] h-fit object-contain'/>
                    <div>
                        <p className='grid-headtext'>My Passion For Coding</p>
                        <p className='grid-subtext'>I am passionate about coding and love to explore new technologies. I am always eager to learn and grow as a developer.</p>
                    </div>
                </div>
            </div>
            <div className='xl:col-span-1 xl:row-span-2'>
                <div className='grid-container '>
                    <img src='/assets/grid4.png' alt="grid-4" className='w-full md:h-[126px] sm:h-[276px] h-fit object-cover sm:object-top'/>
                    <div className='space-y-4'>
                        <p className='grid-subtext text-center'>Contact Me</p>
                        <div className='copy-container' onClick={handleCopy}>
                            <img src={hasCopied?'/assets/tick.svg':'/assets/copy.svg'} alt="copy" className='w-6 h-6'/>
                            <p className='lg-text-2xl md-text-xl font-medium text-gray_gradient text-white'>abhishekkumarpandey2803@gmail.com</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
  )
}

export default About
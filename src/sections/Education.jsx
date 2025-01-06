import React, { Suspense } from 'react'
import { educationBackground } from '../constants'
import Degree from '../components/Degree'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import CanvasLoader from '../components/Loader'

const Education = () => {
  return (
    <section className='c-space my-20' id='education'>
        <div className='w-full text-white-600'>
            <h3 className='head-text'>My Educational Background</h3>
            <div className='work-container'>
                <div className='work-content'>
                    <div className='sm:py-10 py-5 sm:px-5 px-2.5'>
                                            {educationBackground.map(({id,degree, institute, duration, grade, icon})=>(
                                                <div key={id} className='work-content_container group'>
                                                    <div className='flex flex-col h-full justify-start items-center py-2'>
                                                        <div className='work-content_logo'>
                                                            <img src={icon} alt='logo' className='w-full h-full' />
                                                        </div>
                                                        <div className='work-content_bar'/>
                                                    </div>
                                                    <div className='sm-p-5 px-2.5 py-5'>
                                                        <p className='font-bold font-white-800'>{degree}</p>
                                                        <p className='text-sm mb-5'>{institute}--{duration}</p>
                                                        <p className='group-hover:text-white ease-in-out duration-500'>{grade}</p>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                </div>
                <div className='work-canvas'>
                <Canvas>
                        <ambientLight intensity={7}/>
                        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} /> 
                        <directionalLight position={[10,10,10]} intensity={1} />
                        <OrbitControls enableZoom={false} maxPolarAngle={Math.PI / 2}/>
                        <Suspense fallback={<CanvasLoader/>}>
                        <Degree scale={1}/>
                        </Suspense>
                    </Canvas>
                </div>
            </div>
        </div>
    </section>
  )
}

export default Education
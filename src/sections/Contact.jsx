import {useRef, useState} from 'react';
import emailjs from '@emailjs/browser';

const Contact = () => {
    const formRef = useRef();
    const [loading, setLoading] = useState(false);
    const [form, setForm] = useState({
        name: '',
        email: '',
        message: ''
    });
    const handleChange =({target: {name, value}}) => {
        setForm({...form, [name]: value})
    };
    //service_s0f8pwq
    const handleSubmit = async(e) => {
        e.preventDefault();
        setLoading(true);
        try{
        emailjs('service_odhwfnp',
            'template_qkj266o',
        {
            from_name: form.name,
            to_name: 'Abhishek',
            from_email: form.email,
            to_email: 'abhishekakp5580@gmail',
            message: form.message
        },
        'OnzEMIh-lhOoPMLUM')
        setLoading(false);
        alert('Thank you. I will get back to you as soon as possible');
        setForm({
            name: '',
            email: '',
            message: ''
        }); 
    }catch(error){
        setLoading(false);
        alert('Sorry, something went wrong.',error);
        console.log(error);
    }
        
    };
  return (
    <section className='c-space my-20' id='contact'>
        <div className='relative min-h-screen flex flex-col items-center justify-center'>
            <img src='assets/terminal.png' alt='terminal background' className='absolute inset-0 min-h-screen' />
            <div className='contact-container'>
            <h3 className='head-text'>Let's Connect</h3>
            <p className='text-white-600 text-lg mt-3'>You are looking for a developer or have a project in mind? Let's connect and create something amazing together.</p>
            <form ref={formRef} onSubmit={handleSubmit} className='flex flex-col mt-12 space-y-7'>
                <label className='space-y-3'>
                    <span className='field-label'>Full Name</span>
                    <input 
                    type="text"
                    name='name'
                    value={form.name}
                    placeholder='John Doe'
                    onChange={handleChange}
                    required
                    className='field-input'/>
                </label>
                <label className='space-y-3'>
                    <span className='field-label'>Email</span>
                    <input 
                    type="email"
                    name='email'
                    value={form.email}
                    placeholder='johndoe@gmail.com'
                    onChange={handleChange}
                    required
                    className='field-input'/>
                </label>
                <label className='space-y-3'>
                    <span className='field-label'>Your message</span>
                    <textarea 
                    name='message'
                    value={form.message }
                    rows={5}
                    placeholder='Hi, I wanna give u a job...'
                    onChange={handleChange}
                    required
                    className='field-input'/>
                </label>
                <button className='field-btn' type='submit' disabled={loading}>{loading ? 'Sending...' : 'Send Message'}
                    <img src='assets/arrow-up.png' alt='arrow' className='field-btn_arrow'/></button>
            </form>
            </div>
        </div>
    </section>
  )
}

export default Contact
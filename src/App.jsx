import React from 'react'
import Navbar from './sections/Navbar'
import Hero from './sections/Hero'
import About from './sections/About'
import Projects from './sections/Projects'
import Client from './sections/Client'
import Contact from './sections/Contact'
import Footer from './sections/Footer'
import Experience from './sections/Experience'
import Education from './sections/Education'
import ChatBotWidget from './sections/ChatBotWidget'

const App = () => {
  return (
    <main className="max-w-7xl mx-auto">
      <Navbar />
      {/* <ChatBotWidget /> */}
      <Hero />
      <About />
      <Projects />
      <Client />
      <Experience />
      <Education />
      <Contact />
      <Footer />
    </main>
  )
}

export default App
export const navLinks = [
    {
      id: 1,
      name: 'Home',
      href: '#home',
    },
    {
      id: 2,
      name: 'About',
      href: '#about',
    },
    {
      id: 3,
      name: 'Work',
      href: '#work',
    },
    {
      id: 4,
      name: 'Contact',
      href: '#contact',
    },
  ];
  
  export const clientReviews = [
    {
      id: 1,
      name: 'Ashu Sharma',
      position: 'Senior Software Engineer at Upfront Healthcare',
      img: 'assets/review3.png',
      review:
        'I had the pleasure of working with Abhishek Kumar Pandey at Upfront Healthcare, and I can confidently say that he is an exceptional talent in the realm of software development. His dedication to work is unparalleled. He not only takes responsibility for his tasks but consistently thinks through every possible scenario to ensure the best outcomes.',
    },
    {
      id: 3,
      name: 'Anchal Sharma',
      position: 'Software Engineer at Upfront Healthcare ',
      img: 'assets/review1.png',
      review:
        `One of Abhishek's standout qualities is his proficiency in both backend and frontend development. He possesses a rare blend of technical skill and creativity, making him a valuable asset in any project. His problem-solving abilities are remarkable; he has a unique knack for devising optimized solutions that enhance project performance significantly. This skill set has proven invaluable in handling complex projects, where he has always delivered results that exceed expectations.`,
    },
    {
      id: 2,
      name: 'Ujjwal Sethi',
      position: 'Project Manager at Upfront Healthcare',
      img: 'assets/review2.png',
      review:
        `Abhishek is a brilliant, dedicated professional whose contributions have been a driving force in our team's success. His ability to handle both backend and frontend challenges with innovative and optimized solutions is exceptional. I have no doubt that he will continue to excel and be a valuable asset to any team.`,
    },
    {
      id: 4,
      name: 'Vijaita Kashyap',
      position: 'Professor at Chitkara University',
      img: 'assets/review4.png',
      review:
        'Abhishek is an excellent communicator, able to articulate complex ideas clearly and effectively. This, combined with his fast learning curve and brilliant research capabilities, means he can implement new, cutting-edge technologies and strategies with impressive speed and efficiency. Which is a huge plus for any organization.',
    },
  ];
  
  export const myProjects = [
    {
      title: 'Podcastr - AI Podcast Platform',
      desc: 'Podcastr is a revolutionary Software-as-a-Service platform that transforms the way podcasts are created. With advanced AI-powered features like text-to-multiple-voices functionality, it allows creators to generate diverse voiceovers from a single text input.',
      subdesc:
        'Built as a unique Software-as-a-Service app with Next.js 14, Tailwind CSS, TypeScript, Framer Motion and Convex, Podcastr is designed for optimal performance and scalability.',
      href: 'https://github.com/ABHISHEK-AKP',
      texture: '/textures/project/project1.mp4',
      logo: '/assets/project-logo1.png',
      logoStyle: {
        backgroundColor: '#2A1816',
        border: '0.2px solid #36201D',
        boxShadow: '0px 0px 60px 0px #AA3C304D',
      },
      spotlight: '/assets/spotlight1.png',
      tags: [
        {
          id: 1,
          name: 'React.js',
          path: '/assets/react.svg',
        },
        {
          id: 2,
          name: 'TailwindCSS',
          path: 'assets/tailwindcss.png',
        },
        {
          id: 3,
          name: 'TypeScript',
          path: '/assets/typescript.png',
        },
        {
          id: 4,
          name: 'Framer Motion',
          path: '/assets/framer.png',
        },
      ],
    },
    {
      title: 'Smart Traffic Light Controller',
      desc: '•	Created an AI-driven traffic management system integrating Python, Scikit-Fuzzy, and SUMO.Containerized the system with Docker for reproducible simulation environments.Implemented deterministic decision-making algorithms for IoT sensors to optimize routing & load balancing.',
      subdesc:
        'Tech: Python, Scikit-Fuzzy, SUMO, IoT, Bash, Docker',
      href: 'https://github.com/ABHISHEK-AKP',
      texture: '/textures/project/project2.mp4',
      logo: '/assets/project-logo2.png',
      logoStyle: {
        backgroundColor: '#13202F',
        border: '0.2px solid #17293E',
        boxShadow: '0px 0px 60px 0px #2F6DB54D',
      },
      spotlight: '/assets/spotlight2.png',
      tags: [
        {
          id: 1,
          name: 'React.js',
          path: '/assets/react.svg',
        },
        {
          id: 2,
          name: 'TailwindCSS',
          path: 'assets/tailwindcss.png',
        },
        {
          id: 3,
          name: 'TypeScript',
          path: '/assets/typescript.png',
        },
        {
          id: 4,
          name: 'Framer Motion',
          path: '/assets/framer.png',
        },
      ],
    },
    {
      title: 'Movie Recommender System',
      desc: 'Implemented a collaborative filtering model in Python using Pandas, NumPy, and Scikit-learn, achieving 87% precision in recommendations.Designed data pipelines for processing large datasets efficiently.',
      subdesc:
        '•	Skills Used: Python, Pandas, NumPy, Scikit-learn, Data Processing, Machine Learning',
      href: 'https://github.com/ABHISHEK-AKP',
      texture: '/textures/project/project3.mp4',
      logo: '/assets/project-logo3.png',
      logoStyle: {
        backgroundColor: '#60f5a1',
        background:
          'linear-gradient(0deg, #60F5A150, #60F5A150), linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(208, 213, 221, 0.8) 100%)',
        border: '0.2px solid rgba(208, 213, 221, 1)',
        boxShadow: '0px 0px 60px 0px rgba(35, 131, 96, 0.3)',
      },
      spotlight: '/assets/spotlight3.png',
      ags: [
        {
          id: 1,
          name: 'React.js',
          path: '/assets/react.svg',
        },
        {
          id: 2,
          name: 'TailwindCSS',
          path: 'assets/tailwindcss.png',
        },
        {
          id: 3,
          name: 'TypeScript',
          path: '/assets/typescript.png',
        },
        {
          id: 4,
          name: 'Framer Motion',
          path: '/assets/framer.png',
        },
      ],
    }
  ];
  
  export const calculateSizes = (isSmall, isMobile, isTablet) => {
    return {
      deskScale: isSmall ? 0.05 : isMobile ? 0.06 : 0.065,
      deskPosition: isMobile ? [0.5, -4.5, 0] : [0.25, -5.5, 0],
      cubePosition: isSmall ? [2, -6.5, 0] : isMobile ? [3, -6, 0] : isTablet ? [7, -5, 0] : [9, -5.5, 0],
      reactLogoPosition: isSmall ? [-15, 4, 0] : isMobile ? [-50, 4, 0] : isTablet ? [0, 4, 0] : [12, 3, 0],
      ringPosition: isSmall ? [-5, 7, 0] : isMobile ? [-8.5, 10, 0] : isTablet ? [-14, 10, 0] : [-24, 10, 0],
      targetPosition: isSmall ? [-5, -10, -10] : isMobile ? [-5, -10, -10] : isTablet ? [-11, -9, -10] : [-13, -10, -10],
    };
  };
  
  export const workExperiences = [
    {
      id: 1,
      name: 'Upfront Healthcare',
      pos: 'Software Engineer',
      duration: 'June,2023 - July,2024',
      title: "Full stack developer with diverse project exposure. Developed innovative code for diverse software projects. Integrated third-party apps enhancing software features and performance. Leveraged research and data to address project requirements and challenges. Built a strong rapport with seniors and mentors.",
      icon: '/assets/upfront_healthcare_logo.jpeg',
      animation: 'victory',
    },
    {
      id: 2,
      name: 'Upfront Healthcare',
      pos: 'Software Engineer Intern',
      duration: 'April,2022 - May,2023',
      title: "Performed root cause analysis, fixed defects. Maximized learning opportunities in software engineering. Demonstrated proficiency in crafting MySQL queries, contributed to effective database management and optimization. ",
     icon: '/assets/upfront_healthcare_logo.jpeg',
      animation: 'clapping',
    },
    {
      id: 3,
      name: 'PEC MART',
      pos: 'Founder',
      duration: 'January,2020 - December,2022',
      title: "Created and executed internet ordering platform leading to 20% order completion rate gain. Built and managed a team of 5 in support for marketing & sales, which led to using social media techniques that saw an increase by 40% approx. Built strategic relationships with student clubs and organizations, resulting in high-volume order placements; increased revenue during peak periods by 25%.",
      icon: '/assets/PEClogo.png',
      animation: 'salute',
    },
  ];
  export const educationBackground = [
    {
      id: 1,
      degree: 'Master Of Engineering in Computer Engineering ',
      institute: 'University Of Guelph',
      duration: 'September,2024 - August,2025',
      grade: 'CGPA: 3.6/4.0',
      icon: '/assets/degree-cap.png',
    },
    {
      id: 2,
      degree: 'Bachelor Of Engineering in Computer Science Engineering',
      institute: 'Chitkara University',
      duration: 'August,2019 - May,2023',
      grade: 'CGPA: 9.69/10.0',
      icon: '/assets/study-program.png',
    },
    {
      id: 3,
      degree: '12 th Grade',
      institute: 'Goverment Model Senior Secondary School Sec-16, Chandigarh',
      duration: '2018 - 2019',
      grade: 'Percentage: 72%',
      icon: '/assets/artificial-intelligence.png',
    },
    {
      id: 4,
      degree: '10 th Grade',
      institute: 'Manav Mangal School',
      duration: '2016 - 2017',
      grade: 'CGPA: 10.0/10.0',
      icon: '/assets/open-book.png',
    }
  ];
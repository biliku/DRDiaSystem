import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Layout from '../views/Layout.vue'
import UserManagement from '../views/modules/UserManagement.vue'
import DataManagement from '../views/modules/DataManagement.vue'
import ImageProcessing from '../views/modules/ImageProcessing.vue'
import Dashboard from '../views/modules/Dashboard.vue'

const routes = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/login',
    name: 'Login',
    component: Login
  },
  {
    path: '/layout',
    name: 'Layout',
    component: Layout,
    children: [
      {
        path: '/user-management',
        name: 'UserManagement',
        component: UserManagement
      },
      {
        path: '/data-management',
        name: 'DataManagement',
        component: DataManagement
      },
      {
        path: '/image-processing',
        name: 'ImageProcessing',
        component: ImageProcessing
      },
      {
        path: '/dashboard',
        name: 'Dashboard',
        component: Dashboard
      }
      // 其他子路由可以在这里添加
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 全局前置守卫
router.beforeEach((to, from, next) => {
  // 如果是登录页面，直接通过
  if (to.path === '/login') {
    next()
    return
  }
  
  // 检查是否有token
  const token = localStorage.getItem('token')
  if (!token) {
    next('/login')
    return
  }
  
  next()
})

export default router 
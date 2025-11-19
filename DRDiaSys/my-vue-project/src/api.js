import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000' })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

let isRefreshing = false
let pending = []
function onRefreshed(newToken) {
  pending.forEach(cb => cb(newToken))
  pending = []
}

api.interceptors.response.use(
  res => res,
  async err => {
    const { response, config } = err || {}
    if (!response || response.status !== 401 || config._retry) throw err

    const refresh = localStorage.getItem('refresh')
    if (!refresh) {
      localStorage.clear()
      window.location.href = '/login'
      throw err
    }

    if (isRefreshing) {
      return new Promise(resolve => {
        pending.push((newToken) => {
          config.headers.Authorization = `Bearer ${newToken}`
          config._retry = true
          resolve(api(config))
        })
      })
    }

    isRefreshing = true
    try {
      const { data } = await axios.post('http://localhost:8000/api/token/refresh/', { refresh })
      localStorage.setItem('token', data.access)
      onRefreshed(data.access)
      config.headers.Authorization = `Bearer ${data.access}`
      config._retry = true
      return api(config)
    } catch (e) {
      localStorage.clear()
      window.location.href = '/login'
      throw e
    } finally {
      isRefreshing = false
    }
  }
)

export default api



